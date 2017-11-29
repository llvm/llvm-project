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

namespace {
// Helper for mapping JSON objects onto our protocol structs. Intended use:
// Optional<Result> parse(json::Expr E) {
//   ObjectParser O(E);
//   if (!O || !O.parse("mandatory_field", Result.MandatoryField))
//     return None;
//   O.parse("optional_field", Result.OptionalField);
//   return Result;
// }
// FIXME: the static methods here should probably become the public parse()
// extension point. Overloading free functions allows us to uniformly handle
// enums, vectors, etc.
class ObjectParser {
public:
  ObjectParser(const json::Expr &E) : O(E.asObject()) {}

  // True if the expression is an object.
  operator bool() { return O; }

  template <typename T> bool parse(const char *Prop, T &Out) {
    assert(*this && "Must check this is an object before calling parse()");
    if (const json::Expr *E = O->get(Prop))
      return parse(*E, Out);
    return false;
  }

  // Optional requires special handling, because missing keys are OK.
  template <typename T> bool parse(const char *Prop, llvm::Optional<T> &Out) {
    assert(*this && "Must check this is an object before calling parse()");
    if (const json::Expr *E = O->get(Prop))
      return parse(*E, Out);
    Out = None;
    return true;
  }

private:
  // Primitives.
  static bool parse(const json::Expr &E, std::string &Out) {
    if (auto S = E.asString()) {
      Out = *S;
      return true;
    }
    return false;
  }

  static bool parse(const json::Expr &E, int &Out) {
    if (auto S = E.asInteger()) {
      Out = *S;
      return true;
    }
    return false;
  }

  static bool parse(const json::Expr &E, bool &Out) {
    if (auto S = E.asBoolean()) {
      Out = *S;
      return true;
    }
    return false;
  }

  // Types with a parse() function.
  template <typename T> static bool parse(const json::Expr &E, T &Out) {
    if (auto Parsed = std::remove_reference<T>::type::parse(E)) {
      Out = std::move(*Parsed);
      return true;
    }
    return false;
  }

  // Nullable values as Optional<T>.
  template <typename T>
  static bool parse(const json::Expr &E, llvm::Optional<T> &Out) {
    if (E.asNull()) {
      Out = None;
      return true;
    }
    T Result;
    if (!parse(E, Result))
      return false;
    Out = std::move(Result);
    return true;
  }

  // Array values with std::vector type.
  template <typename T>
  static bool parse(const json::Expr &E, std::vector<T> &Out) {
    if (auto *A = E.asArray()) {
      Out.clear();
      Out.resize(A->size());
      for (size_t I = 0; I < A->size(); ++I)
        if (!parse((*A)[I], Out[I]))
          return false;
      return true;
    }
    return false;
  }

  // Object values with std::map<std::string, ?>
  template <typename T>
  static bool parse(const json::Expr &E, std::map<std::string, T> &Out) {
    if (auto *O = E.asObject()) {
      for (const auto &KV : *O)
        if (!parse(KV.second, Out[StringRef(KV.first)]))
          return false;
      return true;
    }
    return false;
  }

  // Special cased enums, which can't have T::parse() functions.
  // FIXME: make everything free functions so there's no special casing.
  static bool parse(const json::Expr &E, TraceLevel &Out) {
    if (auto S = E.asString()) {
      if (*S == "off") {
        Out = TraceLevel::Off;
        return true;
      } else if (*S == "messages") {
        Out = TraceLevel::Messages;
        return true;
      } else if (*S == "verbose") {
        Out = TraceLevel::Verbose;
        return true;
      }
    }
    return false;
  }

  static bool parse(const json::Expr &E, FileChangeType &Out) {
    if (auto T = E.asInteger()) {
      if (*T < static_cast<int>(FileChangeType::Created) ||
          *T > static_cast<int>(FileChangeType::Deleted))
        return false;
      Out = static_cast<FileChangeType>(*T);
      return true;
    }
    return false;
  }

  const json::obj *O;
};
} // namespace

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

llvm::Optional<URI> URI::parse(const json::Expr &E) {
  if (auto S = E.asString())
    return fromUri(*S);
  return None;
}

json::Expr URI::unparse(const URI &U) { return U.uri; }

llvm::Optional<TextDocumentIdentifier>
TextDocumentIdentifier::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  TextDocumentIdentifier R;
  if (!O || !O.parse("uri", R.uri))
    return None;
  return R;
}

llvm::Optional<Position> Position::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  Position R;
  if (!O || !O.parse("line", R.line) || !O.parse("character", R.character))
    return None;
  return R;
}

json::Expr Position::unparse(const Position &P) {
  return json::obj{
      {"line", P.line},
      {"character", P.character},
  };
}

llvm::Optional<Range> Range::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  Range R;
  if (!O || !O.parse("start", R.start) || !O.parse("end", R.end))
    return None;
  return R;
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
  ObjectParser O(Params);
  TextDocumentItem R;
  if (!O || !O.parse("uri", R.uri) || !O.parse("languageId", R.languageId) ||
      !O.parse("version", R.version) || !O.parse("text", R.text))
    return None;
  return R;
}

llvm::Optional<Metadata> Metadata::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  Metadata R;
  if (!O)
    return None;
  O.parse("extraFlags", R.extraFlags);
  return R;
}

llvm::Optional<TextEdit> TextEdit::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  TextEdit R;
  if (!O || !O.parse("range", R.range) || !O.parse("newText", R.newText))
    return None;
  return R;
}

json::Expr TextEdit::unparse(const TextEdit &P) {
  return json::obj{
      {"range", P.range},
      {"newText", P.newText},
  };
}

llvm::Optional<InitializeParams>
InitializeParams::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  InitializeParams R;
  if (!O)
    return None;
  // We deliberately don't fail if we can't parse individual fields.
  // Failing to handle a slightly malformed initialize would be a disaster.
  O.parse("processId", R.processId);
  O.parse("rootUri", R.rootUri);
  O.parse("rootPath", R.rootPath);
  O.parse("trace", R.trace);
  // initializationOptions, capabilities unused
  return R;
}

llvm::Optional<DidOpenTextDocumentParams>
DidOpenTextDocumentParams::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  DidOpenTextDocumentParams R;
  if (!O || !O.parse("textDocument", R.textDocument) ||
      !O.parse("metadata", R.metadata))
    return None;
  return R;
}

llvm::Optional<DidCloseTextDocumentParams>
DidCloseTextDocumentParams::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  DidCloseTextDocumentParams R;
  if (!O || !O.parse("textDocument", R.textDocument))
    return None;
  return R;
}

llvm::Optional<DidChangeTextDocumentParams>
DidChangeTextDocumentParams::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  DidChangeTextDocumentParams R;
  if (!O || !O.parse("textDocument", R.textDocument) ||
      !O.parse("contentChanges", R.contentChanges))
    return None;
  return R;
}

llvm::Optional<FileEvent> FileEvent::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  FileEvent R;
  if (!O || !O.parse("uri", R.uri) || !O.parse("type", R.type))
    return None;
  return R;
}

llvm::Optional<DidChangeWatchedFilesParams>
DidChangeWatchedFilesParams::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  DidChangeWatchedFilesParams R;
  if (!O || !O.parse("changes", R.changes))
    return None;
  return R;
}

llvm::Optional<TextDocumentContentChangeEvent>
TextDocumentContentChangeEvent::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  TextDocumentContentChangeEvent R;
  if (!O || !O.parse("text", R.text))
    return None;
  return R;
}

llvm::Optional<FormattingOptions>
FormattingOptions::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  FormattingOptions R;
  if (!O || !O.parse("tabSize", R.tabSize) ||
      !O.parse("insertSpaces", R.insertSpaces))
    return None;
  return R;
}

json::Expr FormattingOptions::unparse(const FormattingOptions &P) {
  return json::obj{
      {"tabSize", P.tabSize},
      {"insertSpaces", P.insertSpaces},
  };
}

llvm::Optional<DocumentRangeFormattingParams>
DocumentRangeFormattingParams::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  DocumentRangeFormattingParams R;
  if (!O || !O.parse("textDocument", R.textDocument) ||
      !O.parse("range", R.range) || !O.parse("options", R.options))
    return None;
  return R;
}

llvm::Optional<DocumentOnTypeFormattingParams>
DocumentOnTypeFormattingParams::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  DocumentOnTypeFormattingParams R;
  if (!O || !O.parse("textDocument", R.textDocument) ||
      !O.parse("position", R.position) || !O.parse("ch", R.ch) ||
      !O.parse("options", R.options))
    return None;
  return R;
}

llvm::Optional<DocumentFormattingParams>
DocumentFormattingParams::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  DocumentFormattingParams R;
  if (!O || !O.parse("textDocument", R.textDocument) ||
      !O.parse("options", R.options))
    return None;
  return R;
}

llvm::Optional<Diagnostic> Diagnostic::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  Diagnostic R;
  if (!O || !O.parse("range", R.range) || !O.parse("message", R.message))
    return None;
  O.parse("severity", R.severity);
  return R;
}

llvm::Optional<CodeActionContext>
CodeActionContext::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  CodeActionContext R;
  if (!O || !O.parse("diagnostics", R.diagnostics))
    return None;
  return R;
}

llvm::Optional<CodeActionParams>
CodeActionParams::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  CodeActionParams R;
  if (!O || !O.parse("textDocument", R.textDocument) ||
      !O.parse("range", R.range) || !O.parse("context", R.context))
    return None;
  return R;
}

llvm::Optional<WorkspaceEdit> WorkspaceEdit::parse(const json::Expr &Params) {
  ObjectParser O(Params);
  WorkspaceEdit R;
  if (!O || !O.parse("changes", R.changes))
    return None;
  return R;
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
  ObjectParser O(Params);
  TextDocumentPositionParams R;
  if (!O || !O.parse("textDocument", R.textDocument) ||
      !O.parse("position", R.position))
    return None;
  return R;
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
  ObjectParser O(Params);
  RenameParams R;
  if (!O || !O.parse("textDocument", R.textDocument) ||
      !O.parse("position", R.position) || !O.parse("newName", R.newName))
    return None;
  return R;
}
