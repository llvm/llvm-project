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
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "URI.h"
#include "Logger.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {

bool fromJSON(const json::Expr &E, URIForFile &R) {
  if (auto S = E.asString()) {
    auto U = URI::parse(*S);
    if (!U) {
      log("Failed to parse URI " + *S + ": " + llvm::toString(U.takeError()));
      return false;
    }
    if (U->scheme() != "file") {
      log("Clangd only supports 'file' URI scheme for workspace files: " + *S);
      return false;
    }
    auto Path = URI::resolve(*U);
    if (!Path) {
      log(llvm::toString(Path.takeError()));
      return false;
    }
    R.file = *Path;
    return true;
  }
  return false;
}

json::Expr toJSON(const URIForFile &U) {
  return U.uri();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const URIForFile &U) {
  return OS << U.uri();
}

bool fromJSON(const json::Expr &Params, TextDocumentIdentifier &R) {
  json::ObjectMapper O(Params);
  return O && O.map("uri", R.uri);
}

bool fromJSON(const json::Expr &Params, Position &R) {
  json::ObjectMapper O(Params);
  return O && O.map("line", R.line) && O.map("character", R.character);
}

json::Expr toJSON(const Position &P) {
  return json::obj{
      {"line", P.line},
      {"character", P.character},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Position &P) {
  return OS << P.line << ':' << P.character;
}

bool fromJSON(const json::Expr &Params, Range &R) {
  json::ObjectMapper O(Params);
  return O && O.map("start", R.start) && O.map("end", R.end);
}

json::Expr toJSON(const Range &P) {
  return json::obj{
      {"start", P.start},
      {"end", P.end},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Range &R) {
  return OS << R.start << '-' << R.end;
}

json::Expr toJSON(const Location &P) {
  return json::obj{
      {"uri", P.uri},
      {"range", P.range},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Location &L) {
  return OS << L.range << '@' << L.uri;
}

bool fromJSON(const json::Expr &Params, TextDocumentItem &R) {
  json::ObjectMapper O(Params);
  return O && O.map("uri", R.uri) && O.map("languageId", R.languageId) &&
         O.map("version", R.version) && O.map("text", R.text);
}

bool fromJSON(const json::Expr &Params, Metadata &R) {
  json::ObjectMapper O(Params);
  if (!O)
    return false;
  O.map("extraFlags", R.extraFlags);
  return true;
}

bool fromJSON(const json::Expr &Params, TextEdit &R) {
  json::ObjectMapper O(Params);
  return O && O.map("range", R.range) && O.map("newText", R.newText);
}

json::Expr toJSON(const TextEdit &P) {
  return json::obj{
      {"range", P.range},
      {"newText", P.newText},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const TextEdit &TE) {
  OS << TE.range << " => \"";
  PrintEscapedString(TE.newText, OS);
  return OS << '"';
}

bool fromJSON(const json::Expr &E, TraceLevel &Out) {
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

bool fromJSON(const json::Expr &Params, InitializeParams &R) {
  json::ObjectMapper O(Params);
  if (!O)
    return false;
  // We deliberately don't fail if we can't parse individual fields.
  // Failing to handle a slightly malformed initialize would be a disaster.
  O.map("processId", R.processId);
  O.map("rootUri", R.rootUri);
  O.map("rootPath", R.rootPath);
  O.map("trace", R.trace);
  // initializationOptions, capabilities unused
  return true;
}

bool fromJSON(const json::Expr &Params, DidOpenTextDocumentParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("metadata", R.metadata);
}

bool fromJSON(const json::Expr &Params, DidCloseTextDocumentParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const json::Expr &Params, DidChangeTextDocumentParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("contentChanges", R.contentChanges);
}

bool fromJSON(const json::Expr &E, FileChangeType &Out) {
  if (auto T = E.asInteger()) {
    if (*T < static_cast<int>(FileChangeType::Created) ||
        *T > static_cast<int>(FileChangeType::Deleted))
      return false;
    Out = static_cast<FileChangeType>(*T);
    return true;
  }
  return false;
}

bool fromJSON(const json::Expr &Params, FileEvent &R) {
  json::ObjectMapper O(Params);
  return O && O.map("uri", R.uri) && O.map("type", R.type);
}

bool fromJSON(const json::Expr &Params, DidChangeWatchedFilesParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("changes", R.changes);
}

bool fromJSON(const json::Expr &Params, TextDocumentContentChangeEvent &R) {
  json::ObjectMapper O(Params);
  return O && O.map("text", R.text);
}

bool fromJSON(const json::Expr &Params, FormattingOptions &R) {
  json::ObjectMapper O(Params);
  return O && O.map("tabSize", R.tabSize) &&
         O.map("insertSpaces", R.insertSpaces);
}

json::Expr toJSON(const FormattingOptions &P) {
  return json::obj{
      {"tabSize", P.tabSize},
      {"insertSpaces", P.insertSpaces},
  };
}

bool fromJSON(const json::Expr &Params, DocumentRangeFormattingParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("range", R.range) && O.map("options", R.options);
}

bool fromJSON(const json::Expr &Params, DocumentOnTypeFormattingParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("position", R.position) && O.map("ch", R.ch) &&
         O.map("options", R.options);
}

bool fromJSON(const json::Expr &Params, DocumentFormattingParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("options", R.options);
}

bool fromJSON(const json::Expr &Params, Diagnostic &R) {
  json::ObjectMapper O(Params);
  if (!O || !O.map("range", R.range) || !O.map("message", R.message))
    return false;
  O.map("severity", R.severity);
  return true;
}

bool fromJSON(const json::Expr &Params, CodeActionContext &R) {
  json::ObjectMapper O(Params);
  return O && O.map("diagnostics", R.diagnostics);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Diagnostic &D) {
  OS << D.range << " [";
  switch (D.severity) {
    case 1:
      OS << "error";
      break;
    case 2:
      OS << "warning";
      break;
    case 3:
      OS << "note";
      break;
    case 4:
      OS << "remark";
      break;
    default:
      OS << "diagnostic";
      break;
  }
  return OS << '(' << D.severity << "): " << D.message << "]";
}

bool fromJSON(const json::Expr &Params, CodeActionParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("range", R.range) && O.map("context", R.context);
}

bool fromJSON(const json::Expr &Params, WorkspaceEdit &R) {
  json::ObjectMapper O(Params);
  return O && O.map("changes", R.changes);
}

const llvm::StringLiteral ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND =
    "clangd.applyFix";

bool fromJSON(const json::Expr &Params, ExecuteCommandParams &R) {
  json::ObjectMapper O(Params);
  if (!O || !O.map("command", R.command))
    return false;

  auto Args = Params.asObject()->getArray("arguments");
  if (R.command == ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND) {
    return Args && Args->size() == 1 &&
           fromJSON(Args->front(), R.workspaceEdit);
  }
  return false; // Unrecognized command.
}

json::Expr toJSON(const WorkspaceEdit &WE) {
  if (!WE.changes)
    return json::obj{};
  json::obj FileChanges;
  for (auto &Change : *WE.changes)
    FileChanges[Change.first] = json::ary(Change.second);
  return json::obj{{"changes", std::move(FileChanges)}};
}

json::Expr toJSON(const ApplyWorkspaceEditParams &Params) {
  return json::obj{{"edit", Params.edit}};
}

bool fromJSON(const json::Expr &Params, TextDocumentPositionParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("position", R.position);
}

json::Expr toJSON(const CompletionItem &CI) {
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

bool operator<(const CompletionItem &L, const CompletionItem &R) {
  return (L.sortText.empty() ? L.label : L.sortText) <
         (R.sortText.empty() ? R.label : R.sortText);
}

json::Expr toJSON(const CompletionList &L) {
  return json::obj{
      {"isIncomplete", L.isIncomplete},
      {"items", json::ary(L.items)},
  };
}

json::Expr toJSON(const ParameterInformation &PI) {
  assert(!PI.label.empty() && "parameter information label is required");
  json::obj Result{{"label", PI.label}};
  if (!PI.documentation.empty())
    Result["documentation"] = PI.documentation;
  return std::move(Result);
}

json::Expr toJSON(const SignatureInformation &SI) {
  assert(!SI.label.empty() && "signature information label is required");
  json::obj Result{
      {"label", SI.label},
      {"parameters", json::ary(SI.parameters)},
  };
  if (!SI.documentation.empty())
    Result["documentation"] = SI.documentation;
  return std::move(Result);
}

json::Expr toJSON(const SignatureHelp &SH) {
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

bool fromJSON(const json::Expr &Params, RenameParams &R) {
  json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("position", R.position) && O.map("newName", R.newName);
}

json::Expr toJSON(const DocumentHighlight &DH) {
  return json::obj{
      {"range", toJSON(DH.range)},
      {"kind", static_cast<int>(DH.kind)},
  };
}

} // namespace clangd
} // namespace clang
