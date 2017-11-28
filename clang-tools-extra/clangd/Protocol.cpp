//===--- Protocol.cpp - Language Server Protocol Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the serialization code for the LSP structs.
// FIXME: This is extremely repetetive and ugly. Is there a better way?
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::clangd;

URI URI::fromUri(llvm::StringRef uri) {
  URI Result;
  Result.uri = uri;
  uri.consume_front("file://");
  // Also trim authority-less URIs
  uri.consume_front("file:");
  // For Windows paths e.g. /X:
  if (uri.size() > 2 && uri[0] == '/' && uri[2] == ':')
    uri.consume_front("/");
  // Make sure that file paths are in native separators
  Result.file = llvm::sys::path::convert_to_slash(uri);
  return Result;
}

URI URI::fromFile(llvm::StringRef file) {
  using namespace llvm::sys;
  URI Result;
  Result.file = file;
  Result.uri = "file://";
  // For Windows paths e.g. X:
  if (file.size() > 1 && file[1] == ':')
    Result.uri += "/";
  // Make sure that uri paths are with posix separators
  Result.uri += path::convert_to_slash(file, path::Style::posix);
  return Result;
}

json::Expr URI::unparse(const URI &U) { return U.uri; }

llvm::Optional<TextDocumentIdentifier>
TextDocumentIdentifier::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  TextDocumentIdentifier Result;
  if (auto U = O->getString("uri"))
    Result.uri = URI::parse(*U);
  // FIXME: parse 'version', but only for VersionedTextDocumentIdentifiers.
  return Result;
}

llvm::Optional<Position> Position::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  Position Result;
  if (auto L = O->getInteger("line"))
    Result.line = *L;
  if (auto C = O->getInteger("character"))
    Result.character = *C;
  return Result;
}

json::Expr Position::unparse(const Position &P) {
  return json::obj{
      {"line", P.line},
      {"character", P.character},
  };
}

llvm::Optional<Range> Range::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  Range Result;
  if (auto *S = O->get("start")) {
    if (auto P = Position::parse(*S))
      Result.start = std::move(*P);
    else
      return None;
  }
  if (auto *E = O->get("end")) {
    if (auto P = Position::parse(*E))
      Result.end = std::move(*P);
    else
      return None;
  }
  return Result;
}

json::Expr Range::unparse(const Range &P) {
  return json::obj{
      {"start", P.start},
      {"end", P.end},
  };
}

json::Expr Location::unparse(const Location &P) {
  return json::obj{
      {"uri", P.uri},
      {"range", P.range},
  };
}

llvm::Optional<TextDocumentItem>
TextDocumentItem::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  TextDocumentItem Result;
  if (auto U = O->getString("uri"))
    Result.uri = URI::parse(*U);
  if (auto L = O->getString("languageId"))
    Result.languageId = *L;
  if (auto V = O->getInteger("version"))
    Result.version = *V;
  if (auto T = O->getString("text"))
    Result.text = *T;
  return Result;
}

llvm::Optional<Metadata> Metadata::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  Metadata Result;
  if (auto *Flags = O->getArray("extraFlags"))
    for (auto &F : *Flags) {
      if (auto S = F.asString())
        Result.extraFlags.push_back(*S);
      else
        return llvm::None;
    }
  return Result;
}

llvm::Optional<TextEdit> TextEdit::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  TextEdit Result;
  if (auto *R = O->get("range")) {
    if (auto Parsed = Range::parse(*R))
      Result.range = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto T = O->getString("newText"))
    Result.newText = *T;
  return Result;
}

json::Expr TextEdit::unparse(const TextEdit &P) {
  return json::obj{
      {"range", P.range},
      {"newText", P.newText},
  };
}

namespace {
TraceLevel getTraceLevel(llvm::StringRef TraceLevelStr) {
  if (TraceLevelStr == "off")
    return TraceLevel::Off;
  else if (TraceLevelStr == "messages")
    return TraceLevel::Messages;
  else if (TraceLevelStr == "verbose")
    return TraceLevel::Verbose;
  return TraceLevel::Off;
}
} // namespace

llvm::Optional<InitializeParams>
InitializeParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  InitializeParams Result;
  if (auto P = O->getInteger("processId"))
    Result.processId = *P;
  if (auto R = O->getString("rootPath"))
    Result.rootPath = *R;
  if (auto R = O->getString("rootUri"))
    Result.rootUri = URI::parse(*R);
  if (auto T = O->getString("trace"))
    Result.trace = getTraceLevel(*T);
  // initializationOptions, capabilities unused
  return Result;
}

llvm::Optional<DidOpenTextDocumentParams>
DidOpenTextDocumentParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  DidOpenTextDocumentParams Result;
  if (auto *D = O->get("textDocument")) {
    if (auto Parsed = TextDocumentItem::parse(*D))
      Result.textDocument = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *M = O->get("metadata")) {
    if (auto Parsed = Metadata::parse(*M))
      Result.metadata = std::move(*Parsed);
    else
      return llvm::None;
  }
  return Result;
}

llvm::Optional<DidCloseTextDocumentParams>
DidCloseTextDocumentParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  DidCloseTextDocumentParams Result;
  if (auto *D = O->get("textDocument")) {
    if (auto Parsed = TextDocumentIdentifier::parse(*D))
      Result.textDocument = std::move(*Parsed);
    else
      return llvm::None;
  }
  return Result;
}

llvm::Optional<DidChangeTextDocumentParams>
DidChangeTextDocumentParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  DidChangeTextDocumentParams Result;
  if (auto *D = O->get("textDocument")) {
    if (auto Parsed = TextDocumentIdentifier::parse(*D))
      Result.textDocument = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *A = O->getArray("contentChanges"))
    for (auto &E : *A) {
      if (auto Parsed = TextDocumentContentChangeEvent::parse(E))
        Result.contentChanges.push_back(std::move(*Parsed));
      else
        return llvm::None;
    }
  return Result;
}

llvm::Optional<FileEvent> FileEvent::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  FileEvent Result;
  if (auto U = O->getString("uri"))
    Result.uri = URI::parse(*U);
  if (auto T = O->getInteger("type")) {
    if (*T < static_cast<int>(FileChangeType::Created) ||
        *T > static_cast<int>(FileChangeType::Deleted))
      return llvm::None;
    Result.type = static_cast<FileChangeType>(*T);
  }
  return Result;
}

llvm::Optional<DidChangeWatchedFilesParams>
DidChangeWatchedFilesParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  DidChangeWatchedFilesParams Result;
  if (auto *C = O->getArray("changes"))
    for (auto &E : *C) {
      if (auto Parsed = FileEvent::parse(E))
        Result.changes.push_back(std::move(*Parsed));
      else
        return llvm::None;
    }
  return Result;
}

llvm::Optional<TextDocumentContentChangeEvent>
TextDocumentContentChangeEvent::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  TextDocumentContentChangeEvent Result;
  if (auto T = O->getString("text"))
    Result.text = *T;
  return Result;
}

llvm::Optional<FormattingOptions>
FormattingOptions::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  FormattingOptions Result;
  if (auto T = O->getInteger("tabSize"))
    Result.tabSize = *T;
  if (auto I = O->getBoolean("insertSpaces"))
    Result.insertSpaces = *I;
  return Result;
}

json::Expr FormattingOptions::unparse(const FormattingOptions &P) {
  return json::obj{
      {"tabSize", P.tabSize},
      {"insertSpaces", P.insertSpaces},
  };
}

llvm::Optional<DocumentRangeFormattingParams>
DocumentRangeFormattingParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  DocumentRangeFormattingParams Result;
  if (auto *D = O->get("textDocument")) {
    if (auto Parsed = TextDocumentIdentifier::parse(*D))
      Result.textDocument = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *R = O->get("range")) {
    if (auto Parsed = Range::parse(*R))
      Result.range = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *F = O->get("options")) {
    if (auto Parsed = FormattingOptions::parse(*F))
      Result.options = std::move(*Parsed);
    else
      return llvm::None;
  }
  return Result;
}

llvm::Optional<DocumentOnTypeFormattingParams>
DocumentOnTypeFormattingParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  DocumentOnTypeFormattingParams Result;
  if (auto Ch = O->getString("ch"))
    Result.ch = *Ch;
  if (auto *D = O->get("textDocument")) {
    if (auto Parsed = TextDocumentIdentifier::parse(*D))
      Result.textDocument = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *P = O->get("position")) {
    if (auto Parsed = Position::parse(*P))
      Result.position = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *F = O->get("options")) {
    if (auto Parsed = FormattingOptions::parse(*F))
      Result.options = std::move(*Parsed);
    else
      return llvm::None;
  }
  return Result;
}

llvm::Optional<DocumentFormattingParams>
DocumentFormattingParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  DocumentFormattingParams Result;
  if (auto *D = O->get("textDocument")) {
    if (auto Parsed = TextDocumentIdentifier::parse(*D))
      Result.textDocument = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *F = O->get("options")) {
    if (auto Parsed = FormattingOptions::parse(*F))
      Result.options = std::move(*Parsed);
    else
      return llvm::None;
  }
  return Result;
}

llvm::Optional<Diagnostic> Diagnostic::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  Diagnostic Result;
  if (auto *R = O->get("range")) {
    if (auto Parsed = Range::parse(*R))
      Result.range = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto S = O->getInteger("severity"))
    Result.severity = *S;
  if (auto M = O->getString("message"))
    Result.message = *M;
  return Result;
}

llvm::Optional<CodeActionContext>
CodeActionContext::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  CodeActionContext Result;
  if (auto *D = O->getArray("diagnostics"))
    for (auto &E : *D) {
      if (auto Parsed = Diagnostic::parse(E))
        Result.diagnostics.push_back(std::move(*Parsed));
      else
        return llvm::None;
    }
  return Result;
}

llvm::Optional<CodeActionParams>
CodeActionParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  CodeActionParams Result;
  if (auto *D = O->get("textDocument")) {
    if (auto Parsed = TextDocumentIdentifier::parse(*D))
      Result.textDocument = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *R = O->get("range")) {
    if (auto Parsed = Range::parse(*R))
      Result.range = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *R = O->get("context")) {
    if (auto Parsed = CodeActionContext::parse(*R))
      Result.context = std::move(*Parsed);
    else
      return llvm::None;
  }
  return Result;
}

llvm::Optional<std::map<std::string, std::vector<TextEdit>>>
parseWorkspaceEditChange(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  std::map<std::string, std::vector<TextEdit>> Result;
  for (const auto &KV : *O) {
    auto &Values = Result[StringRef(KV.first)];
    if (auto *Edits = KV.second.asArray())
      for (auto &Edit : *Edits) {
        if (auto Parsed = TextEdit::parse(Edit))
          Values.push_back(std::move(*Parsed));
        else
          return llvm::None;
      }
    else
      return llvm::None;
  }
  return Result;
}

llvm::Optional<WorkspaceEdit> WorkspaceEdit::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  WorkspaceEdit Result;
  if (auto *C = O->get("changes")) {
    if (auto Parsed = parseWorkspaceEditChange(*C))
      Result.changes = std::move(*Parsed);
    else
      return llvm::None;
  }
  return Result;
}

const std::string ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND =
    "clangd.applyFix";

llvm::Optional<ExecuteCommandParams>
ExecuteCommandParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  ExecuteCommandParams Result;
  if (auto Command = O->getString("command"))
    Result.command = *Command;
  auto Args = O->getArray("arguments");

  if (Result.command == ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND) {
    if (!Args || Args->size() != 1)
      return llvm::None;
    if (auto Parsed = WorkspaceEdit::parse(Args->front()))
      Result.workspaceEdit = std::move(*Parsed);
    else
      return llvm::None;
  } else
    return llvm::None; // Unrecognized command.
  return Result;
}

json::Expr WorkspaceEdit::unparse(const WorkspaceEdit &WE) {
  if (!WE.changes)
    return json::obj{};
  json::obj FileChanges;
  for (auto &Change : *WE.changes)
    FileChanges[Change.first] = json::ary(Change.second);
  return json::obj{{"changes", std::move(FileChanges)}};
}

json::Expr
ApplyWorkspaceEditParams::unparse(const ApplyWorkspaceEditParams &Params) {
  return json::obj{{"edit", Params.edit}};
}

llvm::Optional<TextDocumentPositionParams>
TextDocumentPositionParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  TextDocumentPositionParams Result;
  if (auto *D = O->get("textDocument")) {
    if (auto Parsed = TextDocumentIdentifier::parse(*D))
      Result.textDocument = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *P = O->get("position")) {
    if (auto Parsed = Position::parse(*P))
      Result.position = std::move(*Parsed);
    else
      return llvm::None;
  }
  return Result;
}

json::Expr CompletionItem::unparse(const CompletionItem &CI) {
  assert(!CI.label.empty() && "completion item label is required");
  json::obj Result{{"label", CI.label}};
  if (CI.kind != CompletionItemKind::Missing)
    Result["kind"] = static_cast<int>(CI.kind);
  if (!CI.detail.empty())
    Result["detail"] = CI.detail;
  if (!CI.documentation.empty())
    Result["documentation"] = CI.documentation;
  if (!CI.sortText.empty())
    Result["sortText"] = CI.sortText;
  if (!CI.filterText.empty())
    Result["filterText"] = CI.filterText;
  if (!CI.insertText.empty())
    Result["insertText"] = CI.insertText;
  if (CI.insertTextFormat != InsertTextFormat::Missing)
    Result["insertTextFormat"] = static_cast<int>(CI.insertTextFormat);
  if (CI.textEdit)
    Result["textEdit"] = *CI.textEdit;
  if (!CI.additionalTextEdits.empty())
    Result["additionalTextEdits"] = json::ary(CI.additionalTextEdits);
  return std::move(Result);
}

bool clangd::operator<(const CompletionItem &L, const CompletionItem &R) {
  return (L.sortText.empty() ? L.label : L.sortText) <
         (R.sortText.empty() ? R.label : R.sortText);
}

json::Expr CompletionList::unparse(const CompletionList &L) {
  return json::obj{
      {"isIncomplete", L.isIncomplete},
      {"items", json::ary(L.items)},
  };
}

json::Expr ParameterInformation::unparse(const ParameterInformation &PI) {
  assert(!PI.label.empty() && "parameter information label is required");
  json::obj Result{{"label", PI.label}};
  if (!PI.documentation.empty())
    Result["documentation"] = PI.documentation;
  return std::move(Result);
}

json::Expr SignatureInformation::unparse(const SignatureInformation &SI) {
  assert(!SI.label.empty() && "signature information label is required");
  json::obj Result{
      {"label", SI.label},
      {"parameters", json::ary(SI.parameters)},
  };
  if (!SI.documentation.empty())
    Result["documentation"] = SI.documentation;
  return std::move(Result);
}

json::Expr SignatureHelp::unparse(const SignatureHelp &SH) {
  assert(SH.activeSignature >= 0 &&
         "Unexpected negative value for number of active signatures.");
  assert(SH.activeParameter >= 0 &&
         "Unexpected negative value for active parameter index");
  return json::obj{
      {"activeSignature", SH.activeSignature},
      {"activeParameter", SH.activeParameter},
      {"signatures", json::ary(SH.signatures)},
  };
}

llvm::Optional<RenameParams> RenameParams::parse(const json::Expr &Params) {
  const json::obj *O = Params.asObject();
  if (!O)
    return None;

  RenameParams Result;
  if (auto *D = O->get("textDocument")) {
    if (auto Parsed = TextDocumentIdentifier::parse(*D))
      Result.textDocument = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto *P = O->get("position")) {
    if (auto Parsed = Position::parse(*P))
      Result.position = std::move(*Parsed);
    else
      return llvm::None;
  }
  if (auto N = O->getString("newName"))
    Result.newName = *N;
  return Result;
}
