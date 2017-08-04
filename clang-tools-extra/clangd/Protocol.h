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
// Each struct has a parse and unparse function, that converts back and forth
// between the struct and a JSON representation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROTOCOL_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROTOCOL_H

#include "llvm/ADT/Optional.h"
#include "llvm/Support/YAMLParser.h"
#include <string>
#include <vector>

namespace clang {
namespace clangd {

struct URI {
  std::string uri;
  std::string file;

  static URI fromUri(llvm::StringRef uri);
  static URI fromFile(llvm::StringRef file);

  static URI parse(llvm::yaml::ScalarNode *Param);
  static std::string unparse(const URI &U);

  friend bool operator==(const URI &LHS, const URI &RHS) {
    return LHS.uri == RHS.uri;
  }

  friend bool operator!=(const URI &LHS, const URI &RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const URI &LHS, const URI &RHS) {
    return LHS.uri < RHS.uri;
  }
};

struct TextDocumentIdentifier {
  /// The text document's URI.
  URI uri;

  static llvm::Optional<TextDocumentIdentifier>
  parse(llvm::yaml::MappingNode *Params);
};

struct Position {
  /// Line position in a document (zero-based).
  int line;

  /// Character offset on a line in a document (zero-based).
  int character;

  friend bool operator==(const Position &LHS, const Position &RHS) {
    return std::tie(LHS.line, LHS.character) ==
           std::tie(RHS.line, RHS.character);
  }
  friend bool operator<(const Position &LHS, const Position &RHS) {
    return std::tie(LHS.line, LHS.character) <
           std::tie(RHS.line, RHS.character);
  }

  static llvm::Optional<Position> parse(llvm::yaml::MappingNode *Params);
  static std::string unparse(const Position &P);
};

struct Range {
  /// The range's start position.
  Position start;

  /// The range's end position.
  Position end;

  friend bool operator==(const Range &LHS, const Range &RHS) {
    return std::tie(LHS.start, LHS.end) == std::tie(RHS.start, RHS.end);
  }
  friend bool operator<(const Range &LHS, const Range &RHS) {
    return std::tie(LHS.start, LHS.end) < std::tie(RHS.start, RHS.end);
  }

  static llvm::Optional<Range> parse(llvm::yaml::MappingNode *Params);
  static std::string unparse(const Range &P);
};

struct Location {
  /// The text document's URI.
  URI uri;
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

  static std::string unparse(const Location &P);
};

struct Metadata {
  std::vector<std::string> extraFlags;

  static llvm::Optional<Metadata> parse(llvm::yaml::MappingNode *Params);
};

struct TextEdit {
  /// The range of the text document to be manipulated. To insert
  /// text into a document create a range where start === end.
  Range range;

  /// The string to be inserted. For delete operations use an
  /// empty string.
  std::string newText;

  static llvm::Optional<TextEdit> parse(llvm::yaml::MappingNode *Params);
  static std::string unparse(const TextEdit &P);
};

struct TextDocumentItem {
  /// The text document's URI.
  URI uri;

  /// The text document's language identifier.
  std::string languageId;

  /// The version number of this document (it will strictly increase after each
  int version;

  /// The content of the opened text document.
  std::string text;

  static llvm::Optional<TextDocumentItem>
  parse(llvm::yaml::MappingNode *Params);
};

struct DidOpenTextDocumentParams {
  /// The document that was opened.
  TextDocumentItem textDocument;

  /// Extension storing per-file metadata, such as compilation flags.
  llvm::Optional<Metadata> metadata;

  static llvm::Optional<DidOpenTextDocumentParams>
  parse(llvm::yaml::MappingNode *Params);
};

struct DidCloseTextDocumentParams {
  /// The document that was closed.
  TextDocumentIdentifier textDocument;

  static llvm::Optional<DidCloseTextDocumentParams>
  parse(llvm::yaml::MappingNode *Params);
};

struct TextDocumentContentChangeEvent {
  /// The new text of the document.
  std::string text;

  static llvm::Optional<TextDocumentContentChangeEvent>
  parse(llvm::yaml::MappingNode *Params);
};

struct DidChangeTextDocumentParams {
  /// The document that did change. The version number points
  /// to the version after all provided content changes have
  /// been applied.
  TextDocumentIdentifier textDocument;

  /// The actual content changes.
  std::vector<TextDocumentContentChangeEvent> contentChanges;

  static llvm::Optional<DidChangeTextDocumentParams>
  parse(llvm::yaml::MappingNode *Params);
};

struct FormattingOptions {
  /// Size of a tab in spaces.
  int tabSize;

  /// Prefer spaces over tabs.
  bool insertSpaces;

  static llvm::Optional<FormattingOptions>
  parse(llvm::yaml::MappingNode *Params);
  static std::string unparse(const FormattingOptions &P);
};

struct DocumentRangeFormattingParams {
  /// The document to format.
  TextDocumentIdentifier textDocument;

  /// The range to format
  Range range;

  /// The format options
  FormattingOptions options;

  static llvm::Optional<DocumentRangeFormattingParams>
  parse(llvm::yaml::MappingNode *Params);
};

struct DocumentOnTypeFormattingParams {
  /// The document to format.
  TextDocumentIdentifier textDocument;

  /// The position at which this request was sent.
  Position position;

  /// The character that has been typed.
  std::string ch;

  /// The format options.
  FormattingOptions options;

  static llvm::Optional<DocumentOnTypeFormattingParams>
  parse(llvm::yaml::MappingNode *Params);
};

struct DocumentFormattingParams {
  /// The document to format.
  TextDocumentIdentifier textDocument;

  /// The format options
  FormattingOptions options;

  static llvm::Optional<DocumentFormattingParams>
  parse(llvm::yaml::MappingNode *Params);
};

struct Diagnostic {
  /// The range at which the message applies.
  Range range;

  /// The diagnostic's severity. Can be omitted. If omitted it is up to the
  /// client to interpret diagnostics as error, warning, info or hint.
  int severity;

  /// The diagnostic's message.
  std::string message;

  friend bool operator==(const Diagnostic &LHS, const Diagnostic &RHS) {
    return std::tie(LHS.range, LHS.severity, LHS.message) ==
           std::tie(RHS.range, RHS.severity, RHS.message);
  }
  friend bool operator<(const Diagnostic &LHS, const Diagnostic &RHS) {
    return std::tie(LHS.range, LHS.severity, LHS.message) <
           std::tie(RHS.range, RHS.severity, RHS.message);
  }

  static llvm::Optional<Diagnostic> parse(llvm::yaml::MappingNode *Params);
};

struct CodeActionContext {
  /// An array of diagnostics.
  std::vector<Diagnostic> diagnostics;

  static llvm::Optional<CodeActionContext>
  parse(llvm::yaml::MappingNode *Params);
};

struct CodeActionParams {
  /// The document in which the command was invoked.
  TextDocumentIdentifier textDocument;

  /// The range for which the command was invoked.
  Range range;

  /// Context carrying additional information.
  CodeActionContext context;

  static llvm::Optional<CodeActionParams>
  parse(llvm::yaml::MappingNode *Params);
};

struct TextDocumentPositionParams {
  /// The text document.
  TextDocumentIdentifier textDocument;

  /// The position inside the text document.
  Position position;

  static llvm::Optional<TextDocumentPositionParams>
  parse(llvm::yaml::MappingNode *Params);
};

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

  // TODO(krasimir): The following optional fields defined by the language
  // server protocol are unsupported:
  //
  // command?: Command - An optional command that is executed *after* inserting
  //                     this completion.
  //
  // data?: any - A data entry field that is preserved on a completion item
  //              between a completion and a completion resolve request.
  static std::string unparse(const CompletionItem &P);
};

} // namespace clangd
} // namespace clang

#endif
