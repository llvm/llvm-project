//===--- Protocol.cpp - Language Server Protocol Implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the serialization code for the LSP structs.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "Logger.h"
#include "URI.h"
#include "index/Index.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {

char LSPError::ID;

URIForFile URIForFile::canonicalize(llvm::StringRef AbsPath,
                                    llvm::StringRef TUPath) {
  assert(llvm::sys::path::is_absolute(AbsPath) && "the path is relative");
  auto Resolved = URI::resolvePath(AbsPath, TUPath);
  if (!Resolved) {
    elog("URIForFile: failed to resolve path {0} with TU path {1}: "
         "{2}.\nUsing unresolved path.",
         AbsPath, TUPath, Resolved.takeError());
    return URIForFile(AbsPath);
  }
  return URIForFile(std::move(*Resolved));
}

llvm::Expected<URIForFile> URIForFile::fromURI(const URI &U,
                                               llvm::StringRef HintPath) {
  auto Resolved = URI::resolve(U, HintPath);
  if (!Resolved)
    return Resolved.takeError();
  return URIForFile(std::move(*Resolved));
}

bool fromJSON(const llvm::json::Value &E, URIForFile &R) {
  if (auto S = E.getAsString()) {
    auto Parsed = URI::parse(*S);
    if (!Parsed) {
      elog("Failed to parse URI {0}: {1}", *S, Parsed.takeError());
      return false;
    }
    if (Parsed->scheme() != "file" && Parsed->scheme() != "test") {
      elog("Clangd only supports 'file' URI scheme for workspace files: {0}",
           *S);
      return false;
    }
    // "file" and "test" schemes do not require hint path.
    auto U = URIForFile::fromURI(*Parsed, /*HintPath=*/"");
    if (!U) {
      elog("{0}", U.takeError());
      return false;
    }
    R = std::move(*U);
    return true;
  }
  return false;
}

llvm::json::Value toJSON(const URIForFile &U) { return U.uri(); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const URIForFile &U) {
  return OS << U.uri();
}

llvm::json::Value toJSON(const TextDocumentIdentifier &R) {
  return llvm::json::Object{{"uri", R.uri}};
}

bool fromJSON(const llvm::json::Value &Params, TextDocumentIdentifier &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("uri", R.uri);
}

bool fromJSON(const llvm::json::Value &Params, Position &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("line", R.line) && O.map("character", R.character);
}

llvm::json::Value toJSON(const Position &P) {
  return llvm::json::Object{
      {"line", P.line},
      {"character", P.character},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Position &P) {
  return OS << P.line << ':' << P.character;
}

bool fromJSON(const llvm::json::Value &Params, Range &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("start", R.start) && O.map("end", R.end);
}

llvm::json::Value toJSON(const Range &P) {
  return llvm::json::Object{
      {"start", P.start},
      {"end", P.end},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Range &R) {
  return OS << R.start << '-' << R.end;
}

llvm::json::Value toJSON(const Location &P) {
  return llvm::json::Object{
      {"uri", P.uri},
      {"range", P.range},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Location &L) {
  return OS << L.range << '@' << L.uri;
}

bool fromJSON(const llvm::json::Value &Params, TextDocumentItem &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("uri", R.uri) && O.map("languageId", R.languageId) &&
         O.map("version", R.version) && O.map("text", R.text);
}

bool fromJSON(const llvm::json::Value &Params, TextEdit &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("range", R.range) && O.map("newText", R.newText);
}

llvm::json::Value toJSON(const TextEdit &P) {
  return llvm::json::Object{
      {"range", P.range},
      {"newText", P.newText},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const TextEdit &TE) {
  OS << TE.range << " => \"";
  llvm::printEscapedString(TE.newText, OS);
  return OS << '"';
}

bool fromJSON(const llvm::json::Value &E, TraceLevel &Out) {
  if (auto S = E.getAsString()) {
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

bool fromJSON(const llvm::json::Value &E, SymbolKind &Out) {
  if (auto T = E.getAsInteger()) {
    if (*T < static_cast<int>(SymbolKind::File) ||
        *T > static_cast<int>(SymbolKind::TypeParameter))
      return false;
    Out = static_cast<SymbolKind>(*T);
    return true;
  }
  return false;
}

bool fromJSON(const llvm::json::Value &E, SymbolKindBitset &Out) {
  if (auto *A = E.getAsArray()) {
    for (size_t I = 0; I < A->size(); ++I) {
      SymbolKind KindOut;
      if (fromJSON((*A)[I], KindOut))
        Out.set(size_t(KindOut));
    }
    return true;
  }
  return false;
}

SymbolKind adjustKindToCapability(SymbolKind Kind,
                                  SymbolKindBitset &SupportedSymbolKinds) {
  auto KindVal = static_cast<size_t>(Kind);
  if (KindVal >= SymbolKindMin && KindVal <= SupportedSymbolKinds.size() &&
      SupportedSymbolKinds[KindVal])
    return Kind;

  switch (Kind) {
  // Provide some fall backs for common kinds that are close enough.
  case SymbolKind::Struct:
    return SymbolKind::Class;
  case SymbolKind::EnumMember:
    return SymbolKind::Enum;
  default:
    return SymbolKind::String;
  }
}

bool fromJSON(const llvm::json::Value &Params, ClientCapabilities &R) {
  const llvm::json::Object *O = Params.getAsObject();
  if (!O)
    return false;
  if (auto *TextDocument = O->getObject("textDocument")) {
    if (auto *Diagnostics = TextDocument->getObject("publishDiagnostics")) {
      if (auto CategorySupport = Diagnostics->getBoolean("categorySupport"))
        R.DiagnosticCategory = *CategorySupport;
      if (auto CodeActions = Diagnostics->getBoolean("codeActionsInline"))
        R.DiagnosticFixes = *CodeActions;
    }
    if (auto *Completion = TextDocument->getObject("completion")) {
      if (auto *Item = Completion->getObject("completionItem")) {
        if (auto SnippetSupport = Item->getBoolean("snippetSupport"))
          R.CompletionSnippets = *SnippetSupport;
      }
      if (auto *ItemKind = Completion->getObject("completionItemKind")) {
        if (auto *ValueSet = ItemKind->get("valueSet")) {
          R.CompletionItemKinds.emplace();
          if (!fromJSON(*ValueSet, *R.CompletionItemKinds))
            return false;
        }
      }
    }
    if (auto *CodeAction = TextDocument->getObject("codeAction")) {
      if (CodeAction->getObject("codeActionLiteralSupport"))
        R.CodeActionStructure = true;
    }
    if (auto *DocumentSymbol = TextDocument->getObject("documentSymbol")) {
      if (auto HierarchicalSupport =
              DocumentSymbol->getBoolean("hierarchicalDocumentSymbolSupport"))
        R.HierarchicalDocumentSymbol = *HierarchicalSupport;
    }
  }
  if (auto *Workspace = O->getObject("workspace")) {
    if (auto *Symbol = Workspace->getObject("symbol")) {
      if (auto *SymbolKind = Symbol->getObject("symbolKind")) {
        if (auto *ValueSet = SymbolKind->get("valueSet")) {
          R.WorkspaceSymbolKinds.emplace();
          if (!fromJSON(*ValueSet, *R.WorkspaceSymbolKinds))
            return false;
        }
      }
    }
  }
  return true;
}

bool fromJSON(const llvm::json::Value &Params, InitializeParams &R) {
  llvm::json::ObjectMapper O(Params);
  if (!O)
    return false;
  // We deliberately don't fail if we can't parse individual fields.
  // Failing to handle a slightly malformed initialize would be a disaster.
  O.map("processId", R.processId);
  O.map("rootUri", R.rootUri);
  O.map("rootPath", R.rootPath);
  O.map("capabilities", R.capabilities);
  O.map("trace", R.trace);
  O.map("initializationOptions", R.initializationOptions);
  return true;
}

bool fromJSON(const llvm::json::Value &Params, DidOpenTextDocumentParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value &Params, DidCloseTextDocumentParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value &Params, DidChangeTextDocumentParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("contentChanges", R.contentChanges) &&
         O.map("wantDiagnostics", R.wantDiagnostics);
}

bool fromJSON(const llvm::json::Value &E, FileChangeType &Out) {
  if (auto T = E.getAsInteger()) {
    if (*T < static_cast<int>(FileChangeType::Created) ||
        *T > static_cast<int>(FileChangeType::Deleted))
      return false;
    Out = static_cast<FileChangeType>(*T);
    return true;
  }
  return false;
}

bool fromJSON(const llvm::json::Value &Params, FileEvent &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("uri", R.uri) && O.map("type", R.type);
}

bool fromJSON(const llvm::json::Value &Params, DidChangeWatchedFilesParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("changes", R.changes);
}

bool fromJSON(const llvm::json::Value &Params,
              TextDocumentContentChangeEvent &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("range", R.range) && O.map("rangeLength", R.rangeLength) &&
         O.map("text", R.text);
}

bool fromJSON(const llvm::json::Value &Params, FormattingOptions &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("tabSize", R.tabSize) &&
         O.map("insertSpaces", R.insertSpaces);
}

llvm::json::Value toJSON(const FormattingOptions &P) {
  return llvm::json::Object{
      {"tabSize", P.tabSize},
      {"insertSpaces", P.insertSpaces},
  };
}

bool fromJSON(const llvm::json::Value &Params,
              DocumentRangeFormattingParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("range", R.range) && O.map("options", R.options);
}

bool fromJSON(const llvm::json::Value &Params,
              DocumentOnTypeFormattingParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("position", R.position) && O.map("ch", R.ch) &&
         O.map("options", R.options);
}

bool fromJSON(const llvm::json::Value &Params, DocumentFormattingParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("options", R.options);
}

bool fromJSON(const llvm::json::Value &Params, DocumentSymbolParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument);
}

llvm::json::Value toJSON(const Diagnostic &D) {
  llvm::json::Object Diag{
      {"range", D.range},
      {"severity", D.severity},
      {"message", D.message},
  };
  if (D.category)
    Diag["category"] = *D.category;
  if (D.codeActions)
    Diag["codeActions"] = D.codeActions;
  return std::move(Diag);
}

bool fromJSON(const llvm::json::Value &Params, Diagnostic &R) {
  llvm::json::ObjectMapper O(Params);
  if (!O || !O.map("range", R.range) || !O.map("message", R.message))
    return false;
  O.map("severity", R.severity);
  O.map("category", R.category);
  return true;
}

bool fromJSON(const llvm::json::Value &Params, CodeActionContext &R) {
  llvm::json::ObjectMapper O(Params);
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

bool fromJSON(const llvm::json::Value &Params, CodeActionParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("range", R.range) && O.map("context", R.context);
}

bool fromJSON(const llvm::json::Value &Params, WorkspaceEdit &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("changes", R.changes);
}

const llvm::StringLiteral ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND =
    "clangd.applyFix";
bool fromJSON(const llvm::json::Value &Params, ExecuteCommandParams &R) {
  llvm::json::ObjectMapper O(Params);
  if (!O || !O.map("command", R.command))
    return false;

  auto Args = Params.getAsObject()->getArray("arguments");
  if (R.command == ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND) {
    return Args && Args->size() == 1 &&
           fromJSON(Args->front(), R.workspaceEdit);
  }
  return false; // Unrecognized command.
}

llvm::json::Value toJSON(const SymbolInformation &P) {
  return llvm::json::Object{
      {"name", P.name},
      {"kind", static_cast<int>(P.kind)},
      {"location", P.location},
      {"containerName", P.containerName},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &O,
                              const SymbolInformation &SI) {
  O << SI.containerName << "::" << SI.name << " - " << toJSON(SI);
  return O;
}

bool operator==(const SymbolDetails &LHS, const SymbolDetails &RHS) {
  return LHS.name == RHS.name && LHS.containerName == RHS.containerName &&
         LHS.USR == RHS.USR && LHS.ID == RHS.ID;
}

llvm::json::Value toJSON(const SymbolDetails &P) {
  llvm::json::Object Result{{"name", llvm::json::Value(nullptr)},
                            {"containerName", llvm::json::Value(nullptr)},
                            {"usr", llvm::json::Value(nullptr)},
                            {"id", llvm::json::Value(nullptr)}};

  if (!P.name.empty())
    Result["name"] = P.name;

  if (!P.containerName.empty())
    Result["containerName"] = P.containerName;

  if (!P.USR.empty())
    Result["usr"] = P.USR;

  if (P.ID.hasValue())
    Result["id"] = P.ID.getValue().str();

  // Older clang cannot compile 'return Result', even though it is legal.
  return llvm::json::Value(std::move(Result));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &O, const SymbolDetails &S) {
  if (!S.containerName.empty()) {
    O << S.containerName;
    llvm::StringRef ContNameRef;
    if (!ContNameRef.endswith("::")) {
      O << " ";
    }
  }
  O << S.name << " - " << toJSON(S);
  return O;
}

bool fromJSON(const llvm::json::Value &Params, WorkspaceSymbolParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("query", R.query);
}

llvm::json::Value toJSON(const Command &C) {
  auto Cmd = llvm::json::Object{{"title", C.title}, {"command", C.command}};
  if (C.workspaceEdit)
    Cmd["arguments"] = {*C.workspaceEdit};
  return std::move(Cmd);
}

const llvm::StringLiteral CodeAction::QUICKFIX_KIND = "quickfix";

llvm::json::Value toJSON(const CodeAction &CA) {
  auto CodeAction = llvm::json::Object{{"title", CA.title}};
  if (CA.kind)
    CodeAction["kind"] = *CA.kind;
  if (CA.diagnostics)
    CodeAction["diagnostics"] = llvm::json::Array(*CA.diagnostics);
  if (CA.edit)
    CodeAction["edit"] = *CA.edit;
  if (CA.command)
    CodeAction["command"] = *CA.command;
  return std::move(CodeAction);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &O, const DocumentSymbol &S) {
  return O << S.name << " - " << toJSON(S);
}

llvm::json::Value toJSON(const DocumentSymbol &S) {
  llvm::json::Object Result{{"name", S.name},
                            {"kind", static_cast<int>(S.kind)},
                            {"range", S.range},
                            {"selectionRange", S.selectionRange}};

  if (!S.detail.empty())
    Result["detail"] = S.detail;
  if (!S.children.empty())
    Result["children"] = S.children;
  if (S.deprecated)
    Result["deprecated"] = true;
  // Older gcc cannot compile 'return Result', even though it is legal.
  return llvm::json::Value(std::move(Result));
}

llvm::json::Value toJSON(const WorkspaceEdit &WE) {
  if (!WE.changes)
    return llvm::json::Object{};
  llvm::json::Object FileChanges;
  for (auto &Change : *WE.changes)
    FileChanges[Change.first] = llvm::json::Array(Change.second);
  return llvm::json::Object{{"changes", std::move(FileChanges)}};
}

llvm::json::Value toJSON(const ApplyWorkspaceEditParams &Params) {
  return llvm::json::Object{{"edit", Params.edit}};
}

bool fromJSON(const llvm::json::Value &Params, TextDocumentPositionParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("position", R.position);
}

bool fromJSON(const llvm::json::Value &Params, CompletionContext &R) {
  llvm::json::ObjectMapper O(Params);
  if (!O)
    return false;

  int TriggerKind;
  if (!O.map("triggerKind", TriggerKind))
    return false;
  R.triggerKind = static_cast<CompletionTriggerKind>(TriggerKind);

  if (auto *TC = Params.getAsObject()->get("triggerCharacter"))
    return fromJSON(*TC, R.triggerCharacter);
  return true;
}

bool fromJSON(const llvm::json::Value &Params, CompletionParams &R) {
  if (!fromJSON(Params, static_cast<TextDocumentPositionParams &>(R)))
    return false;
  if (auto *Context = Params.getAsObject()->get("context"))
    return fromJSON(*Context, R.context);
  return true;
}

static llvm::StringRef toTextKind(MarkupKind Kind) {
  switch (Kind) {
  case MarkupKind::PlainText:
    return "plaintext";
  case MarkupKind::Markdown:
    return "markdown";
  }
  llvm_unreachable("Invalid MarkupKind");
}

llvm::json::Value toJSON(const MarkupContent &MC) {
  if (MC.value.empty())
    return nullptr;

  return llvm::json::Object{
      {"kind", toTextKind(MC.kind)},
      {"value", MC.value},
  };
}

llvm::json::Value toJSON(const Hover &H) {
  llvm::json::Object Result{{"contents", toJSON(H.contents)}};

  if (H.range.hasValue())
    Result["range"] = toJSON(*H.range);

  return std::move(Result);
}

bool fromJSON(const llvm::json::Value &E, CompletionItemKind &Out) {
  if (auto T = E.getAsInteger()) {
    if (*T < static_cast<int>(CompletionItemKind::Text) ||
        *T > static_cast<int>(CompletionItemKind::TypeParameter))
      return false;
    Out = static_cast<CompletionItemKind>(*T);
    return true;
  }
  return false;
}

CompletionItemKind
adjustKindToCapability(CompletionItemKind Kind,
                       CompletionItemKindBitset &SupportedCompletionItemKinds) {
  auto KindVal = static_cast<size_t>(Kind);
  if (KindVal >= CompletionItemKindMin &&
      KindVal <= SupportedCompletionItemKinds.size() &&
      SupportedCompletionItemKinds[KindVal])
    return Kind;

  switch (Kind) {
  // Provide some fall backs for common kinds that are close enough.
  case CompletionItemKind::Folder:
    return CompletionItemKind::File;
  case CompletionItemKind::EnumMember:
    return CompletionItemKind::Enum;
  case CompletionItemKind::Struct:
    return CompletionItemKind::Class;
  default:
    return CompletionItemKind::Text;
  }
}

bool fromJSON(const llvm::json::Value &E, CompletionItemKindBitset &Out) {
  if (auto *A = E.getAsArray()) {
    for (size_t I = 0; I < A->size(); ++I) {
      CompletionItemKind KindOut;
      if (fromJSON((*A)[I], KindOut))
        Out.set(size_t(KindOut));
    }
    return true;
  }
  return false;
}

llvm::json::Value toJSON(const CompletionItem &CI) {
  assert(!CI.label.empty() && "completion item label is required");
  llvm::json::Object Result{{"label", CI.label}};
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
    Result["additionalTextEdits"] = llvm::json::Array(CI.additionalTextEdits);
  if (CI.deprecated)
    Result["deprecated"] = CI.deprecated;
  return std::move(Result);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &O, const CompletionItem &I) {
  O << I.label << " - " << toJSON(I);
  return O;
}

bool operator<(const CompletionItem &L, const CompletionItem &R) {
  return (L.sortText.empty() ? L.label : L.sortText) <
         (R.sortText.empty() ? R.label : R.sortText);
}

llvm::json::Value toJSON(const CompletionList &L) {
  return llvm::json::Object{
      {"isIncomplete", L.isIncomplete},
      {"items", llvm::json::Array(L.items)},
  };
}

llvm::json::Value toJSON(const ParameterInformation &PI) {
  assert(!PI.label.empty() && "parameter information label is required");
  llvm::json::Object Result{{"label", PI.label}};
  if (!PI.documentation.empty())
    Result["documentation"] = PI.documentation;
  return std::move(Result);
}

llvm::json::Value toJSON(const SignatureInformation &SI) {
  assert(!SI.label.empty() && "signature information label is required");
  llvm::json::Object Result{
      {"label", SI.label},
      {"parameters", llvm::json::Array(SI.parameters)},
  };
  if (!SI.documentation.empty())
    Result["documentation"] = SI.documentation;
  return std::move(Result);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &O,
                              const SignatureInformation &I) {
  O << I.label << " - " << toJSON(I);
  return O;
}

llvm::json::Value toJSON(const SignatureHelp &SH) {
  assert(SH.activeSignature >= 0 &&
         "Unexpected negative value for number of active signatures.");
  assert(SH.activeParameter >= 0 &&
         "Unexpected negative value for active parameter index");
  return llvm::json::Object{
      {"activeSignature", SH.activeSignature},
      {"activeParameter", SH.activeParameter},
      {"signatures", llvm::json::Array(SH.signatures)},
  };
}

bool fromJSON(const llvm::json::Value &Params, RenameParams &R) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("position", R.position) && O.map("newName", R.newName);
}

llvm::json::Value toJSON(const DocumentHighlight &DH) {
  return llvm::json::Object{
      {"range", toJSON(DH.range)},
      {"kind", static_cast<int>(DH.kind)},
  };
}

llvm::json::Value toJSON(const FileStatus &FStatus) {
  return llvm::json::Object{
      {"uri", FStatus.uri},
      {"state", FStatus.state},
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &O,
                              const DocumentHighlight &V) {
  O << V.range;
  if (V.kind == DocumentHighlightKind::Read)
    O << "(r)";
  if (V.kind == DocumentHighlightKind::Write)
    O << "(w)";
  return O;
}

bool fromJSON(const llvm::json::Value &Params,
              DidChangeConfigurationParams &CCP) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("settings", CCP.settings);
}

bool fromJSON(const llvm::json::Value &Params,
              ClangdCompileCommand &CDbUpdate) {
  llvm::json::ObjectMapper O(Params);
  return O && O.map("workingDirectory", CDbUpdate.workingDirectory) &&
         O.map("compilationCommand", CDbUpdate.compilationCommand);
}

bool fromJSON(const llvm::json::Value &Params, ConfigurationSettings &S) {
  llvm::json::ObjectMapper O(Params);
  if (!O)
    return true; // 'any' type in LSP.
  O.map("compilationDatabaseChanges", S.compilationDatabaseChanges);
  return true;
}

bool fromJSON(const llvm::json::Value &Params, InitializationOptions &Opts) {
  llvm::json::ObjectMapper O(Params);
  if (!O)
    return true; // 'any' type in LSP.

  fromJSON(Params, Opts.ConfigSettings);
  O.map("compilationDatabasePath", Opts.compilationDatabasePath);
  O.map("fallbackFlags", Opts.fallbackFlags);
  O.map("clangdFileStatus", Opts.FileStatus);
  return true;
}

bool fromJSON(const llvm::json::Value &Params, ReferenceParams &R) {
  TextDocumentPositionParams &Base = R;
  return fromJSON(Params, Base);
}

} // namespace clangd
} // namespace clang
