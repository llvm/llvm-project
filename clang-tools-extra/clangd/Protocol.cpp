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
#include "URI.h"
#include "support/Logger.h"
#include "clang/Basic/LLVM.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
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
    return URIForFile(std::string(AbsPath));
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

bool fromJSON(const llvm::json::Value &E, URIForFile &R, llvm::json::Path P) {
  if (auto S = E.getAsString()) {
    auto Parsed = URI::parse(*S);
    if (!Parsed) {
      P.report("failed to parse URI");
      return false;
    }
    if (Parsed->scheme() != "file" && Parsed->scheme() != "test") {
      P.report("clangd only supports 'file' URI scheme for workspace files");
      return false;
    }
    // "file" and "test" schemes do not require hint path.
    auto U = URIForFile::fromURI(*Parsed, /*HintPath=*/"");
    if (!U) {
      P.report("unresolvable URI");
      consumeError(U.takeError());
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

bool fromJSON(const llvm::json::Value &Params, TextDocumentIdentifier &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("uri", R.uri);
}

llvm::json::Value toJSON(const VersionedTextDocumentIdentifier &R) {
  auto Result = toJSON(static_cast<const TextDocumentIdentifier &>(R));
  Result.getAsObject()->try_emplace("version", R.version);
  return Result;
}

bool fromJSON(const llvm::json::Value &Params,
              VersionedTextDocumentIdentifier &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return fromJSON(Params, static_cast<TextDocumentIdentifier &>(R), P) && O &&
         O.map("version", R.version);
}

bool fromJSON(const llvm::json::Value &Params, Position &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
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

bool fromJSON(const llvm::json::Value &Params, Range &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
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

bool fromJSON(const llvm::json::Value &Params, TextDocumentItem &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("uri", R.uri) && O.map("languageId", R.languageId) &&
         O.map("version", R.version) && O.map("text", R.text);
}

bool fromJSON(const llvm::json::Value &Params, TextEdit &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
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

bool fromJSON(const llvm::json::Value &E, TraceLevel &Out, llvm::json::Path P) {
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

bool fromJSON(const llvm::json::Value &E, SymbolKind &Out, llvm::json::Path P) {
  if (auto T = E.getAsInteger()) {
    if (*T < static_cast<int>(SymbolKind::File) ||
        *T > static_cast<int>(SymbolKind::TypeParameter))
      return false;
    Out = static_cast<SymbolKind>(*T);
    return true;
  }
  return false;
}

bool fromJSON(const llvm::json::Value &E, SymbolKindBitset &Out,
              llvm::json::Path P) {
  if (auto *A = E.getAsArray()) {
    for (size_t I = 0; I < A->size(); ++I) {
      SymbolKind KindOut;
      if (fromJSON((*A)[I], KindOut, P.index(I)))
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

SymbolKind indexSymbolKindToSymbolKind(index::SymbolKind Kind) {
  switch (Kind) {
  case index::SymbolKind::Unknown:
    return SymbolKind::Variable;
  case index::SymbolKind::Module:
    return SymbolKind::Module;
  case index::SymbolKind::Namespace:
    return SymbolKind::Namespace;
  case index::SymbolKind::NamespaceAlias:
    return SymbolKind::Namespace;
  case index::SymbolKind::Macro:
    return SymbolKind::String;
  case index::SymbolKind::Enum:
    return SymbolKind::Enum;
  case index::SymbolKind::Struct:
    return SymbolKind::Struct;
  case index::SymbolKind::Class:
    return SymbolKind::Class;
  case index::SymbolKind::Protocol:
    return SymbolKind::Interface;
  case index::SymbolKind::Extension:
    return SymbolKind::Interface;
  case index::SymbolKind::Union:
    return SymbolKind::Class;
  case index::SymbolKind::TypeAlias:
    return SymbolKind::Class;
  case index::SymbolKind::Function:
    return SymbolKind::Function;
  case index::SymbolKind::Variable:
    return SymbolKind::Variable;
  case index::SymbolKind::Field:
    return SymbolKind::Field;
  case index::SymbolKind::EnumConstant:
    return SymbolKind::EnumMember;
  case index::SymbolKind::InstanceMethod:
  case index::SymbolKind::ClassMethod:
  case index::SymbolKind::StaticMethod:
    return SymbolKind::Method;
  case index::SymbolKind::InstanceProperty:
  case index::SymbolKind::ClassProperty:
  case index::SymbolKind::StaticProperty:
    return SymbolKind::Property;
  case index::SymbolKind::Constructor:
  case index::SymbolKind::Destructor:
    return SymbolKind::Constructor;
  case index::SymbolKind::ConversionFunction:
    return SymbolKind::Function;
  case index::SymbolKind::Parameter:
  case index::SymbolKind::NonTypeTemplateParm:
    return SymbolKind::Variable;
  case index::SymbolKind::Using:
    return SymbolKind::Namespace;
  case index::SymbolKind::TemplateTemplateParm:
  case index::SymbolKind::TemplateTypeParm:
    return SymbolKind::TypeParameter;
  }
  llvm_unreachable("invalid symbol kind");
}

bool fromJSON(const llvm::json::Value &Params, ClientCapabilities &R,
              llvm::json::Path P) {
  const llvm::json::Object *O = Params.getAsObject();
  if (!O) {
    P.report("expected object");
    return false;
  }
  if (auto *TextDocument = O->getObject("textDocument")) {
    if (auto *SemanticHighlighting =
            TextDocument->getObject("semanticHighlightingCapabilities")) {
      if (auto SemanticHighlightingSupport =
              SemanticHighlighting->getBoolean("semanticHighlighting"))
        R.TheiaSemanticHighlighting = *SemanticHighlightingSupport;
    }
    if (TextDocument->getObject("semanticTokens"))
      R.SemanticTokens = true;
    if (auto *Diagnostics = TextDocument->getObject("publishDiagnostics")) {
      if (auto CategorySupport = Diagnostics->getBoolean("categorySupport"))
        R.DiagnosticCategory = *CategorySupport;
      if (auto CodeActions = Diagnostics->getBoolean("codeActionsInline"))
        R.DiagnosticFixes = *CodeActions;
      if (auto RelatedInfo = Diagnostics->getBoolean("relatedInformation"))
        R.DiagnosticRelatedInformation = *RelatedInfo;
    }
    if (auto *Completion = TextDocument->getObject("completion")) {
      if (auto *Item = Completion->getObject("completionItem")) {
        if (auto SnippetSupport = Item->getBoolean("snippetSupport"))
          R.CompletionSnippets = *SnippetSupport;
        if (auto DocumentationFormat = Item->getArray("documentationFormat")) {
          for (const auto &Format : *DocumentationFormat) {
            if (fromJSON(Format, R.CompletionDocumentationFormat, P))
              break;
          }
        }
      }
      if (auto *ItemKind = Completion->getObject("completionItemKind")) {
        if (auto *ValueSet = ItemKind->get("valueSet")) {
          R.CompletionItemKinds.emplace();
          if (!fromJSON(*ValueSet, *R.CompletionItemKinds,
                        P.field("textDocument")
                            .field("completion")
                            .field("completionItemKind")
                            .field("valueSet")))
            return false;
        }
      }
      if (auto EditsNearCursor = Completion->getBoolean("editsNearCursor"))
        R.CompletionFixes = *EditsNearCursor;
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
    if (auto *Hover = TextDocument->getObject("hover")) {
      if (auto *ContentFormat = Hover->getArray("contentFormat")) {
        for (const auto &Format : *ContentFormat) {
          if (fromJSON(Format, R.HoverContentFormat, P))
            break;
        }
      }
    }
    if (auto *Help = TextDocument->getObject("signatureHelp")) {
      R.HasSignatureHelp = true;
      if (auto *Info = Help->getObject("signatureInformation")) {
        if (auto *Parameter = Info->getObject("parameterInformation")) {
          if (auto OffsetSupport = Parameter->getBoolean("labelOffsetSupport"))
            R.OffsetsInSignatureHelp = *OffsetSupport;
        }
      }
    }
    if (auto *Rename = TextDocument->getObject("rename")) {
      if (auto RenameSupport = Rename->getBoolean("prepareSupport"))
        R.RenamePrepareSupport = *RenameSupport;
    }
  }
  if (auto *Workspace = O->getObject("workspace")) {
    if (auto *Symbol = Workspace->getObject("symbol")) {
      if (auto *SymbolKind = Symbol->getObject("symbolKind")) {
        if (auto *ValueSet = SymbolKind->get("valueSet")) {
          R.WorkspaceSymbolKinds.emplace();
          if (!fromJSON(*ValueSet, *R.WorkspaceSymbolKinds,
                        P.field("workspace")
                            .field("symbol")
                            .field("symbolKind")
                            .field("valueSet")))
            return false;
        }
      }
    }
  }
  if (auto *Window = O->getObject("window")) {
    if (auto WorkDoneProgress = Window->getBoolean("workDoneProgress"))
      R.WorkDoneProgress = *WorkDoneProgress;
    if (auto Implicit = Window->getBoolean("implicitWorkDoneProgressCreate"))
      R.ImplicitProgressCreation = *Implicit;
  }
  if (auto *OffsetEncoding = O->get("offsetEncoding")) {
    R.offsetEncoding.emplace();
    if (!fromJSON(*OffsetEncoding, *R.offsetEncoding,
                  P.field("offsetEncoding")))
      return false;
  }
  return true;
}

bool fromJSON(const llvm::json::Value &Params, InitializeParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
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

llvm::json::Value toJSON(const WorkDoneProgressCreateParams &P) {
  return llvm::json::Object{{"token", P.token}};
}

llvm::json::Value toJSON(const WorkDoneProgressBegin &P) {
  llvm::json::Object Result{
      {"kind", "begin"},
      {"title", P.title},
  };
  if (P.cancellable)
    Result["cancellable"] = true;
  if (P.percentage)
    Result["percentage"] = 0;

  // FIXME: workaround for older gcc/clang
  return std::move(Result);
}

llvm::json::Value toJSON(const WorkDoneProgressReport &P) {
  llvm::json::Object Result{{"kind", "report"}};
  if (P.cancellable)
    Result["cancellable"] = *P.cancellable;
  if (P.message)
    Result["message"] = *P.message;
  if (P.percentage)
    Result["percentage"] = *P.percentage;
  // FIXME: workaround for older gcc/clang
  return std::move(Result);
}

llvm::json::Value toJSON(const WorkDoneProgressEnd &P) {
  llvm::json::Object Result{{"kind", "end"}};
  if (P.message)
    Result["message"] = *P.message;
  // FIXME: workaround for older gcc/clang
  return std::move(Result);
}

llvm::json::Value toJSON(const MessageType &R) {
  return static_cast<int64_t>(R);
}

llvm::json::Value toJSON(const ShowMessageParams &R) {
  return llvm::json::Object{{"type", R.type}, {"message", R.message}};
}

bool fromJSON(const llvm::json::Value &Params, DidOpenTextDocumentParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value &Params, DidCloseTextDocumentParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value &Params, DidSaveTextDocumentParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value &Params, DidChangeTextDocumentParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("contentChanges", R.contentChanges) &&
         O.map("wantDiagnostics", R.wantDiagnostics) &&
         O.mapOptional("forceRebuild", R.forceRebuild);
}

bool fromJSON(const llvm::json::Value &E, FileChangeType &Out,
              llvm::json::Path P) {
  if (auto T = E.getAsInteger()) {
    if (*T < static_cast<int>(FileChangeType::Created) ||
        *T > static_cast<int>(FileChangeType::Deleted))
      return false;
    Out = static_cast<FileChangeType>(*T);
    return true;
  }
  return false;
}

bool fromJSON(const llvm::json::Value &Params, FileEvent &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("uri", R.uri) && O.map("type", R.type);
}

bool fromJSON(const llvm::json::Value &Params, DidChangeWatchedFilesParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("changes", R.changes);
}

bool fromJSON(const llvm::json::Value &Params,
              TextDocumentContentChangeEvent &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("range", R.range) && O.map("rangeLength", R.rangeLength) &&
         O.map("text", R.text);
}

bool fromJSON(const llvm::json::Value &Params, DocumentRangeFormattingParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument) && O.map("range", R.range);
}

bool fromJSON(const llvm::json::Value &Params,
              DocumentOnTypeFormattingParams &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("position", R.position) && O.map("ch", R.ch);
}

bool fromJSON(const llvm::json::Value &Params, DocumentFormattingParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value &Params, DocumentSymbolParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

llvm::json::Value toJSON(const DiagnosticRelatedInformation &DRI) {
  return llvm::json::Object{
      {"location", DRI.location},
      {"message", DRI.message},
  };
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
  if (!D.code.empty())
    Diag["code"] = D.code;
  if (!D.source.empty())
    Diag["source"] = D.source;
  if (D.relatedInformation)
    Diag["relatedInformation"] = *D.relatedInformation;
  // FIXME: workaround for older gcc/clang
  return std::move(Diag);
}

bool fromJSON(const llvm::json::Value &Params, Diagnostic &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("range", R.range) && O.map("message", R.message) &&
         O.mapOptional("severity", R.severity) &&
         O.mapOptional("category", R.category) &&
         O.mapOptional("code", R.code) && O.mapOptional("source", R.source);
  return true;
}

llvm::json::Value toJSON(const PublishDiagnosticsParams &PDP) {
  llvm::json::Object Result{
      {"uri", PDP.uri},
      {"diagnostics", PDP.diagnostics},
  };
  if (PDP.version)
    Result["version"] = PDP.version;
  return std::move(Result);
}

bool fromJSON(const llvm::json::Value &Params, CodeActionContext &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
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

bool fromJSON(const llvm::json::Value &Params, CodeActionParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("range", R.range) && O.map("context", R.context);
}

bool fromJSON(const llvm::json::Value &Params, WorkspaceEdit &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("changes", R.changes);
}

const llvm::StringLiteral ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND =
    "clangd.applyFix";
const llvm::StringLiteral ExecuteCommandParams::CLANGD_APPLY_TWEAK =
    "clangd.applyTweak";

bool fromJSON(const llvm::json::Value &Params, ExecuteCommandParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  if (!O || !O.map("command", R.command))
    return false;

  auto Args = Params.getAsObject()->getArray("arguments");
  if (R.command == ExecuteCommandParams::CLANGD_APPLY_FIX_COMMAND) {
    return Args && Args->size() == 1 &&
           fromJSON(Args->front(), R.workspaceEdit,
                    P.field("arguments").index(0));
  }
  if (R.command == ExecuteCommandParams::CLANGD_APPLY_TWEAK)
    return Args && Args->size() == 1 &&
           fromJSON(Args->front(), R.tweakArgs, P.field("arguments").index(0));
  return false; // Unrecognized command.
}

llvm::json::Value toJSON(const SymbolInformation &P) {
  llvm::json::Object O{
      {"name", P.name},
      {"kind", static_cast<int>(P.kind)},
      {"location", P.location},
      {"containerName", P.containerName},
  };
  if (P.score)
    O["score"] = *P.score;
  return std::move(O);
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

  // FIXME: workaround for older gcc/clang
  return std::move(Result);
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

bool fromJSON(const llvm::json::Value &Params, WorkspaceSymbolParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("query", R.query);
}

llvm::json::Value toJSON(const Command &C) {
  auto Cmd = llvm::json::Object{{"title", C.title}, {"command", C.command}};
  if (C.workspaceEdit)
    Cmd["arguments"] = {*C.workspaceEdit};
  if (C.tweakArgs)
    Cmd["arguments"] = {*C.tweakArgs};
  return std::move(Cmd);
}

const llvm::StringLiteral CodeAction::QUICKFIX_KIND = "quickfix";
const llvm::StringLiteral CodeAction::REFACTOR_KIND = "refactor";
const llvm::StringLiteral CodeAction::INFO_KIND = "info";

llvm::json::Value toJSON(const CodeAction &CA) {
  auto CodeAction = llvm::json::Object{{"title", CA.title}};
  if (CA.kind)
    CodeAction["kind"] = *CA.kind;
  if (CA.diagnostics)
    CodeAction["diagnostics"] = llvm::json::Array(*CA.diagnostics);
  if (CA.isPreferred)
    CodeAction["isPreferred"] = true;
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
  // FIXME: workaround for older gcc/clang
  return std::move(Result);
}

llvm::json::Value toJSON(const WorkspaceEdit &WE) {
  if (!WE.changes)
    return llvm::json::Object{};
  llvm::json::Object FileChanges;
  for (auto &Change : *WE.changes)
    FileChanges[Change.first] = llvm::json::Array(Change.second);
  return llvm::json::Object{{"changes", std::move(FileChanges)}};
}

bool fromJSON(const llvm::json::Value &Params, TweakArgs &A,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("file", A.file) && O.map("selection", A.selection) &&
         O.map("tweakID", A.tweakID);
}

llvm::json::Value toJSON(const TweakArgs &A) {
  return llvm::json::Object{
      {"tweakID", A.tweakID}, {"selection", A.selection}, {"file", A.file}};
}

llvm::json::Value toJSON(const ApplyWorkspaceEditParams &Params) {
  return llvm::json::Object{{"edit", Params.edit}};
}

bool fromJSON(const llvm::json::Value &Response, ApplyWorkspaceEditResponse &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Response, P);
  return O && O.map("applied", R.applied) &&
         O.map("failureReason", R.failureReason);
}

bool fromJSON(const llvm::json::Value &Params, TextDocumentPositionParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("position", R.position);
}

bool fromJSON(const llvm::json::Value &Params, CompletionContext &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  int TriggerKind;
  if (!O || !O.map("triggerKind", TriggerKind) ||
      !O.mapOptional("triggerCharacter", R.triggerCharacter))
    return false;
  R.triggerKind = static_cast<CompletionTriggerKind>(TriggerKind);
  return true;
}

bool fromJSON(const llvm::json::Value &Params, CompletionParams &R,
              llvm::json::Path P) {
  if (!fromJSON(Params, static_cast<TextDocumentPositionParams &>(R), P))
    return false;
  if (auto *Context = Params.getAsObject()->get("context"))
    return fromJSON(*Context, R.context, P.field("context"));
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

bool fromJSON(const llvm::json::Value &V, MarkupKind &K, llvm::json::Path P) {
  auto Str = V.getAsString();
  if (!Str) {
    P.report("expected string");
    return false;
  }
  if (*Str == "plaintext")
    K = MarkupKind::PlainText;
  else if (*Str == "markdown")
    K = MarkupKind::Markdown;
  else {
    P.report("unknown markup kind");
    return false;
  }
  return true;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, MarkupKind K) {
  return OS << toTextKind(K);
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

bool fromJSON(const llvm::json::Value &E, CompletionItemKind &Out,
              llvm::json::Path P) {
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

bool fromJSON(const llvm::json::Value &E, CompletionItemKindBitset &Out,
              llvm::json::Path P) {
  if (auto *A = E.getAsArray()) {
    for (size_t I = 0; I < A->size(); ++I) {
      CompletionItemKind KindOut;
      if (fromJSON((*A)[I], KindOut, P.index(I)))
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
  if (CI.documentation)
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
  Result["score"] = CI.score;
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
  assert((PI.labelOffsets.hasValue() || !PI.labelString.empty()) &&
         "parameter information label is required");
  llvm::json::Object Result;
  if (PI.labelOffsets)
    Result["label"] =
        llvm::json::Array({PI.labelOffsets->first, PI.labelOffsets->second});
  else
    Result["label"] = PI.labelString;
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

bool fromJSON(const llvm::json::Value &Params, RenameParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
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

constexpr unsigned SemanticTokenEncodingSize = 5;
static llvm::json::Value encodeTokens(llvm::ArrayRef<SemanticToken> Toks) {
  llvm::json::Array Result;
  for (const auto &Tok : Toks) {
    Result.push_back(Tok.deltaLine);
    Result.push_back(Tok.deltaStart);
    Result.push_back(Tok.length);
    Result.push_back(Tok.tokenType);
    Result.push_back(Tok.tokenModifiers);
  }
  assert(Result.size() == SemanticTokenEncodingSize * Toks.size());
  return std::move(Result);
}

bool operator==(const SemanticToken &L, const SemanticToken &R) {
  return std::tie(L.deltaLine, L.deltaStart, L.length, L.tokenType,
                  L.tokenModifiers) == std::tie(R.deltaLine, R.deltaStart,
                                                R.length, R.tokenType,
                                                R.tokenModifiers);
}

llvm::json::Value toJSON(const SemanticTokens &Tokens) {
  return llvm::json::Object{{"resultId", Tokens.resultId},
                            {"data", encodeTokens(Tokens.tokens)}};
}

llvm::json::Value toJSON(const SemanticTokensEdit &Edit) {
  return llvm::json::Object{
      {"start", SemanticTokenEncodingSize * Edit.startToken},
      {"deleteCount", SemanticTokenEncodingSize * Edit.deleteTokens},
      {"data", encodeTokens(Edit.tokens)}};
}

llvm::json::Value toJSON(const SemanticTokensOrDelta &TE) {
  llvm::json::Object Result{{"resultId", TE.resultId}};
  if (TE.edits)
    Result["edits"] = *TE.edits;
  if (TE.tokens)
    Result["data"] = encodeTokens(*TE.tokens);
  return std::move(Result);
}

bool fromJSON(const llvm::json::Value &Params, SemanticTokensParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value &Params, SemanticTokensDeltaParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("previousResultId", R.previousResultId);
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
              DidChangeConfigurationParams &CCP, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("settings", CCP.settings);
}

bool fromJSON(const llvm::json::Value &Params, ClangdCompileCommand &CDbUpdate,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("workingDirectory", CDbUpdate.workingDirectory) &&
         O.map("compilationCommand", CDbUpdate.compilationCommand);
}

bool fromJSON(const llvm::json::Value &Params, ConfigurationSettings &S,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  if (!O)
    return true; // 'any' type in LSP.
  return O.mapOptional("compilationDatabaseChanges",
                       S.compilationDatabaseChanges);
}

bool fromJSON(const llvm::json::Value &Params, InitializationOptions &Opts,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  if (!O)
    return true; // 'any' type in LSP.

  return fromJSON(Params, Opts.ConfigSettings, P) &&
         O.map("compilationDatabasePath", Opts.compilationDatabasePath) &&
         O.mapOptional("fallbackFlags", Opts.fallbackFlags) &&
         O.mapOptional("clangdFileStatus", Opts.FileStatus);
}

bool fromJSON(const llvm::json::Value &E, TypeHierarchyDirection &Out,
              llvm::json::Path P) {
  auto T = E.getAsInteger();
  if (!T)
    return false;
  if (*T < static_cast<int>(TypeHierarchyDirection::Children) ||
      *T > static_cast<int>(TypeHierarchyDirection::Both))
    return false;
  Out = static_cast<TypeHierarchyDirection>(*T);
  return true;
}

bool fromJSON(const llvm::json::Value &Params, TypeHierarchyParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("position", R.position) && O.map("resolve", R.resolve) &&
         O.map("direction", R.direction);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &O,
                              const TypeHierarchyItem &I) {
  return O << I.name << " - " << toJSON(I);
}

llvm::json::Value toJSON(const TypeHierarchyItem &I) {
  llvm::json::Object Result{{"name", I.name},
                            {"kind", static_cast<int>(I.kind)},
                            {"range", I.range},
                            {"selectionRange", I.selectionRange},
                            {"uri", I.uri}};

  if (I.detail)
    Result["detail"] = I.detail;
  if (I.deprecated)
    Result["deprecated"] = I.deprecated;
  if (I.parents)
    Result["parents"] = I.parents;
  if (I.children)
    Result["children"] = I.children;
  if (I.data)
    Result["data"] = I.data;
  return std::move(Result);
}

bool fromJSON(const llvm::json::Value &Params, TypeHierarchyItem &I,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);

  // Required fields.
  return O && O.map("name", I.name) && O.map("kind", I.kind) &&
         O.map("uri", I.uri) && O.map("range", I.range) &&
         O.map("selectionRange", I.selectionRange) &&
         O.mapOptional("detail", I.detail) &&
         O.mapOptional("deprecated", I.deprecated) &&
         O.mapOptional("parents", I.parents) &&
         O.mapOptional("children", I.children) && O.mapOptional("data", I.data);
}

bool fromJSON(const llvm::json::Value &Params,
              ResolveTypeHierarchyItemParams &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("item", R.item) && O.map("resolve", R.resolve) &&
         O.map("direction", R.direction);
}

bool fromJSON(const llvm::json::Value &Params, ReferenceParams &R,
              llvm::json::Path P) {
  TextDocumentPositionParams &Base = R;
  return fromJSON(Params, Base, P);
}

static const char *toString(OffsetEncoding OE) {
  switch (OE) {
  case OffsetEncoding::UTF8:
    return "utf-8";
  case OffsetEncoding::UTF16:
    return "utf-16";
  case OffsetEncoding::UTF32:
    return "utf-32";
  case OffsetEncoding::UnsupportedEncoding:
    return "unknown";
  }
  llvm_unreachable("Unknown clang.clangd.OffsetEncoding");
}
llvm::json::Value toJSON(const OffsetEncoding &OE) { return toString(OE); }
bool fromJSON(const llvm::json::Value &V, OffsetEncoding &OE,
              llvm::json::Path P) {
  auto Str = V.getAsString();
  if (!Str)
    return false;
  OE = llvm::StringSwitch<OffsetEncoding>(*Str)
           .Case("utf-8", OffsetEncoding::UTF8)
           .Case("utf-16", OffsetEncoding::UTF16)
           .Case("utf-32", OffsetEncoding::UTF32)
           .Default(OffsetEncoding::UnsupportedEncoding);
  return true;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, OffsetEncoding Enc) {
  return OS << toString(Enc);
}

bool operator==(const TheiaSemanticHighlightingInformation &Lhs,
                const TheiaSemanticHighlightingInformation &Rhs) {
  return Lhs.Line == Rhs.Line && Lhs.Tokens == Rhs.Tokens;
}

llvm::json::Value
toJSON(const TheiaSemanticHighlightingInformation &Highlighting) {
  return llvm::json::Object{{"line", Highlighting.Line},
                            {"tokens", Highlighting.Tokens},
                            {"isInactive", Highlighting.IsInactive}};
}

llvm::json::Value toJSON(const TheiaSemanticHighlightingParams &Highlighting) {
  return llvm::json::Object{
      {"textDocument", Highlighting.TextDocument},
      {"lines", std::move(Highlighting.Lines)},
  };
}

bool fromJSON(const llvm::json::Value &Params, SelectionRangeParams &S,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", S.textDocument) &&
         O.map("positions", S.positions);
}

llvm::json::Value toJSON(const SelectionRange &Out) {
  if (Out.parent) {
    return llvm::json::Object{{"range", Out.range},
                              {"parent", toJSON(*Out.parent)}};
  }
  return llvm::json::Object{{"range", Out.range}};
}

bool fromJSON(const llvm::json::Value &Params, DocumentLinkParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

llvm::json::Value toJSON(const DocumentLink &DocumentLink) {
  return llvm::json::Object{
      {"range", DocumentLink.range},
      {"target", DocumentLink.target},
  };
}

bool fromJSON(const llvm::json::Value &Params, FoldingRangeParams &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

llvm::json::Value toJSON(const FoldingRange &Range) {
  llvm::json::Object Result{
      {"startLine", Range.startLine},
      {"endLine", Range.endLine},
  };
  if (Range.startCharacter)
    Result["startCharacter"] = Range.startCharacter;
  if (Range.endCharacter)
    Result["endCharacter"] = Range.endCharacter;
  if (Range.kind)
    Result["kind"] = *Range.kind;
  return Result;
}

} // namespace clangd
} // namespace clang
