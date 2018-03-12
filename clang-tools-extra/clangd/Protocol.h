//===--- Protocol.h - Language Server Protocol Implementation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains structs based on the LSP specification at
// https://github.com/Microsoft/language-server-protocol/blob/master/protocol.md
//
// This is not meant to be a complete implementation, new interfaces are added
// when they're needed.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSONExpr.h)
//
// Some structs also have operator<< serialization. This is for debugging and
// tests, and is not generally machine-readable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROTOCOL_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROTOCOL_H

#include "JSONExpr.h"
#include "URI.h"
#include "llvm/ADT/Optional.h"
#include <string>
#include <vector>

namespace clang {
namespace clangd {

enum class ErrorCode {
  // Defined by JSON RPC.
  ParseError = -32700,
  InvalidRequest = -32600,
  MethodNotFound = -32601,
  InvalidParams = -32602,
  InternalError = -32603,

  ServerNotInitialized = -32002,
  UnknownErrorCode = -32001,

  // Defined by the protocol.
  RequestCancelled = -32800,
};

struct URIForFile {
  URIForFile() = default;
  explicit URIForFile(std::string AbsPath);

  /// Retrieves absolute path to the file.
  llvm::StringRef file() const { return File; }

  explicit operator bool() const { return !File.empty(); }
  std::string uri() const { return URI::createFile(File).toString(); }

  friend bool operator==(const URIForFile &LHS, const URIForFile &RHS) {
    return LHS.File == RHS.File;
  }

  friend bool operator!=(const URIForFile &LHS, const URIForFile &RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const URIForFile &LHS, const URIForFile &RHS) {
    return LHS.File < RHS.File;
  }

private:
  std::string File;
};

/// Serialize/deserialize \p URIForFile to/from a string URI.
json::Expr toJSON(const URIForFile &U);
bool fromJSON(const json::Expr &, URIForFile &);

struct TextDocumentIdentifier {
  /// The text document's URI.
  URIForFile uri;
};
json::Expr toJSON(const TextDocumentIdentifier &);
bool fromJSON(const json::Expr &, TextDocumentIdentifier &);

struct Position {
  /// Line position in a document (zero-based).
  int line = 0;

  /// Character offset on a line in a document (zero-based).
  int character = 0;

  friend bool operator==(const Position &LHS, const Position &RHS) {
    return std::tie(LHS.line, LHS.character) ==
           std::tie(RHS.line, RHS.character);
  }
  friend bool operator<(const Position &LHS, const Position &RHS) {
    return std::tie(LHS.line, LHS.character) <
           std::tie(RHS.line, RHS.character);
  }
  friend bool operator<=(const Position &LHS, const Position &RHS) {
    return std::tie(LHS.line, LHS.character) <=
           std::tie(RHS.line, RHS.character);
  }
};
bool fromJSON(const json::Expr &, Position &);
json::Expr toJSON(const Position &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Position &);

struct Range {
  /// The range's start position.
  Position start;

  /// The range's end position.
  Position end;

  friend bool operator==(const Range &LHS, const Range &RHS) {
    return std::tie(LHS.start, LHS.end) == std::tie(RHS.start, RHS.end);
  }
  friend bool operator!=(const Range &LHS, const Range &RHS) {
    return !(LHS == RHS);
  }
  friend bool operator<(const Range &LHS, const Range &RHS) {
    return std::tie(LHS.start, LHS.end) < std::tie(RHS.start, RHS.end);
  }

  bool contains(Position Pos) const { return start <= Pos && Pos < end; }
};
bool fromJSON(const json::Expr &, Range &);
json::Expr toJSON(const Range &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Range &);

struct Location {
  /// The text document's URI.
  URIForFile uri;
  Range range;

  friend bool operator==(const Location &LHS, const Location &RHS) {
    return LHS.uri == RHS.uri && LHS.range == RHS.range;
  }

  friend bool operator!=(const Location &LHS, const Location &RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const Location &LHS, const Location &RHS) {
    return std::tie(LHS.uri, LHS.range) < std::tie(RHS.uri, RHS.range);
  }
};
json::Expr toJSON(const Location &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Location &);

struct Metadata {
  std::vector<std::string> extraFlags;
};
bool fromJSON(const json::Expr &, Metadata &);

struct TextEdit {
  /// The range of the text document to be manipulated. To insert
  /// text into a document create a range where start === end.
  Range range;

  /// The string to be inserted. For delete operations use an
  /// empty string.
  std::string newText;
};
bool fromJSON(const json::Expr &, TextEdit &);
json::Expr toJSON(const TextEdit &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const TextEdit &);

struct TextDocumentItem {
  /// The text document's URI.
  URIForFile uri;

  /// The text document's language identifier.
  std::string languageId;

  /// The version number of this document (it will strictly increase after each
  int version = 0;

  /// The content of the opened text document.
  std::string text;
};
bool fromJSON(const json::Expr &, TextDocumentItem &);

enum class TraceLevel {
  Off = 0,
  Messages = 1,
  Verbose = 2,
};
bool fromJSON(const json::Expr &E, TraceLevel &Out);

struct NoParams {};
inline bool fromJSON(const json::Expr &, NoParams &) { return true; }
using ShutdownParams = NoParams;
using ExitParams = NoParams;

struct CompletionItemClientCapabilities {
  /// Client supports snippets as insert text.
  bool snippetSupport = false;
  /// Client supports commit characters on a completion item.
  bool commitCharacterSupport = false;
  // Client supports the follow content formats for the documentation property.
  // The order describes the preferred format of the client.
  // NOTE: not used by clangd at the moment.
  // std::vector<MarkupKind> documentationFormat;
};
bool fromJSON(const json::Expr &, CompletionItemClientCapabilities &);

struct CompletionClientCapabilities {
  /// Whether completion supports dynamic registration.
  bool dynamicRegistration = false;
  /// The client supports the following `CompletionItem` specific capabilities.
  CompletionItemClientCapabilities completionItem;
  // NOTE: not used by clangd at the moment.
  // llvm::Optional<CompletionItemKindCapabilities> completionItemKind;

  /// The client supports to send additional context information for a
  /// `textDocument/completion` request.
  bool contextSupport = false;
};
bool fromJSON(const json::Expr &, CompletionClientCapabilities &);

// FIXME: most of the capabilities are missing from this struct. Only the ones
// used by clangd are currently there.
struct TextDocumentClientCapabilities {
  /// Capabilities specific to the `textDocument/completion`
  CompletionClientCapabilities completion;
};
bool fromJSON(const json::Expr &, TextDocumentClientCapabilities &);

struct ClientCapabilities {
  // Workspace specific client capabilities.
  // NOTE: not used by clangd at the moment.
  // WorkspaceClientCapabilities workspace;

  // Text document specific client capabilities.
  TextDocumentClientCapabilities textDocument;
};

bool fromJSON(const json::Expr &, ClientCapabilities &);

struct InitializeParams {
  /// The process Id of the parent process that started
  /// the server. Is null if the process has not been started by another
  /// process. If the parent process is not alive then the server should exit
  /// (see exit notification) its process.
  llvm::Optional<int> processId;

  /// The rootPath of the workspace. Is null
  /// if no folder is open.
  ///
  /// @deprecated in favour of rootUri.
  llvm::Optional<std::string> rootPath;

  /// The rootUri of the workspace. Is null if no
  /// folder is open. If both `rootPath` and `rootUri` are set
  /// `rootUri` wins.
  llvm::Optional<URIForFile> rootUri;

  // User provided initialization options.
  // initializationOptions?: any;

  /// The capabilities provided by the client (editor or tool)
  ClientCapabilities capabilities;

  /// The initial trace setting. If omitted trace is disabled ('off').
  llvm::Optional<TraceLevel> trace;
};
bool fromJSON(const json::Expr &, InitializeParams &);

struct DidOpenTextDocumentParams {
  /// The document that was opened.
  TextDocumentItem textDocument;

  /// Extension storing per-file metadata, such as compilation flags.
  llvm::Optional<Metadata> metadata;
};
bool fromJSON(const json::Expr &, DidOpenTextDocumentParams &);

struct DidCloseTextDocumentParams {
  /// The document that was closed.
  TextDocumentIdentifier textDocument;
};
bool fromJSON(const json::Expr &, DidCloseTextDocumentParams &);

struct TextDocumentContentChangeEvent {
  /// The new text of the document.
  std::string text;
};
bool fromJSON(const json::Expr &, TextDocumentContentChangeEvent &);

struct DidChangeTextDocumentParams {
  /// The document that did change. The version number points
  /// to the version after all provided content changes have
  /// been applied.
  TextDocumentIdentifier textDocument;

  /// The actual content changes.
  std::vector<TextDocumentContentChangeEvent> contentChanges;

  /// Forces diagnostics to be generated, or to not be generated, for this
  /// version of the file. If not set, diagnostics are eventually consistent:
  /// either they will be provided for this version or some subsequent one.
  /// This is a clangd extension.
  llvm::Optional<bool> wantDiagnostics;
};
bool fromJSON(const json::Expr &, DidChangeTextDocumentParams &);

enum class FileChangeType {
  /// The file got created.
  Created = 1,
  /// The file got changed.
  Changed = 2,
  /// The file got deleted.
  Deleted = 3
};
bool fromJSON(const json::Expr &E, FileChangeType &Out);

struct FileEvent {
  /// The file's URI.
  URIForFile uri;
  /// The change type.
  FileChangeType type = FileChangeType::Created;
};
bool fromJSON(const json::Expr &, FileEvent &);

struct DidChangeWatchedFilesParams {
  /// The actual file events.
  std::vector<FileEvent> changes;
};
bool fromJSON(const json::Expr &, DidChangeWatchedFilesParams &);

/// Clangd extension to manage a workspace/didChangeConfiguration notification
/// since the data received is described as 'any' type in LSP.
struct ClangdConfigurationParamsChange {
  llvm::Optional<std::string> compilationDatabasePath;
};
bool fromJSON(const json::Expr &, ClangdConfigurationParamsChange &);

struct DidChangeConfigurationParams {
  // We use this predefined struct because it is easier to use
  // than the protocol specified type of 'any'.
  ClangdConfigurationParamsChange settings;
};
bool fromJSON(const json::Expr &, DidChangeConfigurationParams &);

struct FormattingOptions {
  /// Size of a tab in spaces.
  int tabSize = 0;

  /// Prefer spaces over tabs.
  bool insertSpaces = false;
};
bool fromJSON(const json::Expr &, FormattingOptions &);
json::Expr toJSON(const FormattingOptions &);

struct DocumentRangeFormattingParams {
  /// The document to format.
  TextDocumentIdentifier textDocument;

  /// The range to format
  Range range;

  /// The format options
  FormattingOptions options;
};
bool fromJSON(const json::Expr &, DocumentRangeFormattingParams &);

struct DocumentOnTypeFormattingParams {
  /// The document to format.
  TextDocumentIdentifier textDocument;

  /// The position at which this request was sent.
  Position position;

  /// The character that has been typed.
  std::string ch;

  /// The format options.
  FormattingOptions options;
};
bool fromJSON(const json::Expr &, DocumentOnTypeFormattingParams &);

struct DocumentFormattingParams {
  /// The document to format.
  TextDocumentIdentifier textDocument;

  /// The format options
  FormattingOptions options;
};
bool fromJSON(const json::Expr &, DocumentFormattingParams &);

struct Diagnostic {
  /// The range at which the message applies.
  Range range;

  /// The diagnostic's severity. Can be omitted. If omitted it is up to the
  /// client to interpret diagnostics as error, warning, info or hint.
  int severity = 0;

  /// The diagnostic's code. Can be omitted.
  /// Note: Not currently used by clangd
  // std::string code;

  /// A human-readable string describing the source of this
  /// diagnostic, e.g. 'typescript' or 'super lint'.
  /// Note: Not currently used by clangd
  // std::string source;

  /// The diagnostic's message.
  std::string message;
};

/// A LSP-specific comparator used to find diagnostic in a container like
/// std:map.
/// We only use the required fields of Diagnostic to do the comparsion to avoid
/// any regression issues from LSP clients (e.g. VScode), see
/// https://git.io/vbr29
struct LSPDiagnosticCompare {
  bool operator()(const Diagnostic &LHS, const Diagnostic &RHS) const {
    return std::tie(LHS.range, LHS.message) < std::tie(RHS.range, RHS.message);
  }
};
bool fromJSON(const json::Expr &, Diagnostic &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Diagnostic &);

struct CodeActionContext {
  /// An array of diagnostics.
  std::vector<Diagnostic> diagnostics;
};
bool fromJSON(const json::Expr &, CodeActionContext &);

struct CodeActionParams {
  /// The document in which the command was invoked.
  TextDocumentIdentifier textDocument;

  /// The range for which the command was invoked.
  Range range;

  /// Context carrying additional information.
  CodeActionContext context;
};
bool fromJSON(const json::Expr &, CodeActionParams &);

struct WorkspaceEdit {
  /// Holds changes to existing resources.
  llvm::Optional<std::map<std::string, std::vector<TextEdit>>> changes;

  /// Note: "documentChanges" is not currently used because currently there is
  /// no support for versioned edits.
};
bool fromJSON(const json::Expr &, WorkspaceEdit &);
json::Expr toJSON(const WorkspaceEdit &WE);

struct IncludeInsertion {
  /// The document in which the command was invoked.
  /// If either originalHeader or preferredHeader has been (directly) included
  /// in the current file, no new include will be inserted.
  TextDocumentIdentifier textDocument;

  /// The declaring header corresponding to this insertion e.g. the header that
  /// declares a symbol. This could be either a URI or a literal string quoted
  /// with <> or "" that can be #included directly.
  std::string declaringHeader;
  /// The preferred header to be inserted. This may be different from
  /// originalHeader as a header file can have a different canonical include.
  /// This could be either a URI or a literal string quoted with <> or "" that
  /// can be #included directly. If empty, declaringHeader is used to calculate
  /// the #include path.
  std::string preferredHeader;
};
bool fromJSON(const json::Expr &, IncludeInsertion &);
json::Expr toJSON(const IncludeInsertion &II);

/// Exact commands are not specified in the protocol so we define the
/// ones supported by Clangd here. The protocol specifies the command arguments
/// to be "any[]" but to make this safer and more manageable, each command we
/// handle maps to a certain llvm::Optional of some struct to contain its
/// arguments. Different commands could reuse the same llvm::Optional as
/// arguments but a command that needs different arguments would simply add a
/// new llvm::Optional and not use any other ones. In practice this means only
/// one argument type will be parsed and set.
struct ExecuteCommandParams {
  // Command to apply fix-its. Uses WorkspaceEdit as argument.
  const static llvm::StringLiteral CLANGD_APPLY_FIX_COMMAND;
  // Command to insert an #include into code.
  const static llvm::StringLiteral CLANGD_INSERT_HEADER_INCLUDE;

  /// The command identifier, e.g. CLANGD_APPLY_FIX_COMMAND
  std::string command;

  // Arguments
  llvm::Optional<WorkspaceEdit> workspaceEdit;

  llvm::Optional<IncludeInsertion> includeInsertion;
};
bool fromJSON(const json::Expr &, ExecuteCommandParams &);

struct Command : public ExecuteCommandParams {
  std::string title;
};

json::Expr toJSON(const Command &C);

struct ApplyWorkspaceEditParams {
  WorkspaceEdit edit;
};
json::Expr toJSON(const ApplyWorkspaceEditParams &);

struct TextDocumentPositionParams {
  /// The text document.
  TextDocumentIdentifier textDocument;

  /// The position inside the text document.
  Position position;
};
bool fromJSON(const json::Expr &, TextDocumentPositionParams &);

enum class MarkupKind {
  PlainText,
  Markdown,
};

struct MarkupContent {
  MarkupKind kind = MarkupKind::PlainText;
  std::string value;
};
json::Expr toJSON(const MarkupContent &MC);

struct Hover {
  /// The hover's content
  MarkupContent contents;

  /// An optional range is a range inside a text document
  /// that is used to visualize a hover, e.g. by changing the background color.
  llvm::Optional<Range> range;
};
json::Expr toJSON(const Hover &H);

/// The kind of a completion entry.
enum class CompletionItemKind {
  Missing = 0,
  Text = 1,
  Method = 2,
  Function = 3,
  Constructor = 4,
  Field = 5,
  Variable = 6,
  Class = 7,
  Interface = 8,
  Module = 9,
  Property = 10,
  Unit = 11,
  Value = 12,
  Enum = 13,
  Keyword = 14,
  Snippet = 15,
  Color = 16,
  File = 17,
  Reference = 18,
};

/// Defines whether the insert text in a completion item should be interpreted
/// as plain text or a snippet.
enum class InsertTextFormat {
  Missing = 0,
  /// The primary text to be inserted is treated as a plain string.
  PlainText = 1,
  /// The primary text to be inserted is treated as a snippet.
  ///
  /// A snippet can define tab stops and placeholders with `$1`, `$2`
  /// and `${3:foo}`. `$0` defines the final tab stop, it defaults to the end
  /// of the snippet. Placeholders with equal identifiers are linked, that is
  /// typing in one will update others too.
  ///
  /// See also:
  /// https//github.com/Microsoft/vscode/blob/master/src/vs/editor/contrib/snippet/common/snippet.md
  Snippet = 2,
};

/// Provides details for how a completion item was scored.
/// This can be used for client-side filtering of completion items as the
/// user keeps typing.
/// This is a clangd extension.
struct CompletionItemScores {
  /// The score that items are ranked by.
  /// This is filterScore * symbolScore.
  float finalScore = 0.f;
  /// How the partial identifier matched filterText. [0-1]
  float filterScore = 0.f;
  /// How the symbol fits, ignoring the partial identifier.
  float symbolScore = 0.f;
};

struct CompletionItem {
  /// The label of this completion item. By default also the text that is
  /// inserted when selecting this completion.
  std::string label;

  /// The kind of this completion item. Based of the kind an icon is chosen by
  /// the editor.
  CompletionItemKind kind = CompletionItemKind::Missing;

  /// A human-readable string with additional information about this item, like
  /// type or symbol information.
  std::string detail;

  /// A human-readable string that represents a doc-comment.
  std::string documentation;

  /// A string that should be used when comparing this item with other items.
  /// When `falsy` the label is used.
  std::string sortText;

  /// Details about the quality of this completion item. (clangd extension)
  llvm::Optional<CompletionItemScores> scoreInfo;

  /// A string that should be used when filtering a set of completion items.
  /// When `falsy` the label is used.
  std::string filterText;

  /// A string that should be inserted to a document when selecting this
  /// completion. When `falsy` the label is used.
  std::string insertText;

  /// The format of the insert text. The format applies to both the `insertText`
  /// property and the `newText` property of a provided `textEdit`.
  InsertTextFormat insertTextFormat = InsertTextFormat::Missing;

  /// An edit which is applied to a document when selecting this completion.
  /// When an edit is provided `insertText` is ignored.
  ///
  /// Note: The range of the edit must be a single line range and it must
  /// contain the position at which completion has been requested.
  llvm::Optional<TextEdit> textEdit;

  /// An optional array of additional text edits that are applied when selecting
  /// this completion. Edits must not overlap with the main edit nor with
  /// themselves.
  std::vector<TextEdit> additionalTextEdits;

  llvm::Optional<Command> command;
  // TODO(krasimir): The following optional fields defined by the language
  // server protocol are unsupported:
  //
  // data?: any - A data entry field that is preserved on a completion item
  //              between a completion and a completion resolve request.
};
json::Expr toJSON(const CompletionItem &);

bool operator<(const CompletionItem &, const CompletionItem &);

/// Represents a collection of completion items to be presented in the editor.
struct CompletionList {
  /// The list is not complete. Further typing should result in recomputing the
  /// list.
  bool isIncomplete = false;

  /// The completion items.
  std::vector<CompletionItem> items;
};
json::Expr toJSON(const CompletionList &);

/// A single parameter of a particular signature.
struct ParameterInformation {

  /// The label of this parameter. Mandatory.
  std::string label;

  /// The documentation of this parameter. Optional.
  std::string documentation;
};
json::Expr toJSON(const ParameterInformation &);

/// Represents the signature of something callable.
struct SignatureInformation {

  /// The label of this signature. Mandatory.
  std::string label;

  /// The documentation of this signature. Optional.
  std::string documentation;

  /// The parameters of this signature.
  std::vector<ParameterInformation> parameters;
};
json::Expr toJSON(const SignatureInformation &);

/// Represents the signature of a callable.
struct SignatureHelp {

  /// The resulting signatures.
  std::vector<SignatureInformation> signatures;

  /// The active signature.
  int activeSignature = 0;

  /// The active parameter of the active signature.
  int activeParameter = 0;
};
json::Expr toJSON(const SignatureHelp &);

struct RenameParams {
  /// The document that was opened.
  TextDocumentIdentifier textDocument;

  /// The position at which this request was sent.
  Position position;

  /// The new name of the symbol.
  std::string newName;
};
bool fromJSON(const json::Expr &, RenameParams &);

enum class DocumentHighlightKind { Text = 1, Read = 2, Write = 3 };

/// A document highlight is a range inside a text document which deserves
/// special attention. Usually a document highlight is visualized by changing
/// the background color of its range.

struct DocumentHighlight {
  /// The range this highlight applies to.
  Range range;

  /// The highlight kind, default is DocumentHighlightKind.Text.
  DocumentHighlightKind kind = DocumentHighlightKind::Text;

  friend bool operator<(const DocumentHighlight &LHS,
                        const DocumentHighlight &RHS) {
    int LHSKind = static_cast<int>(LHS.kind);
    int RHSKind = static_cast<int>(RHS.kind);
    return std::tie(LHS.range, LHSKind) < std::tie(RHS.range, RHSKind);
  }

  friend bool operator==(const DocumentHighlight &LHS,
                         const DocumentHighlight &RHS) {
    return LHS.kind == RHS.kind && LHS.range == RHS.range;
  }
};
json::Expr toJSON(const DocumentHighlight &DH);

} // namespace clangd
} // namespace clang

#endif
