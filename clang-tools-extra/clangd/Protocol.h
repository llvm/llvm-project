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

struct TextDocumentIdentifier {
  /// The text document's URI.
  std::string uri;

  static llvm::Optional<TextDocumentIdentifier>
  parse(llvm::yaml::MappingNode *Params);
};

struct Position {
  /// Line position in a document (zero-based).
  int line;

  /// Character offset on a line in a document (zero-based).
  int character;

  static llvm::Optional<Position> parse(llvm::yaml::MappingNode *Params);
  static std::string unparse(const Position &P);
};

struct Range {
  /// The range's start position.
  Position start;

  /// The range's end position.
  Position end;

  static llvm::Optional<Range> parse(llvm::yaml::MappingNode *Params);
  static std::string unparse(const Range &P);
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
  std::string uri;

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

  static llvm::Optional<DidOpenTextDocumentParams>
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

struct DocumentFormattingParams {
  /// The document to format.
  TextDocumentIdentifier textDocument;

  /// The format options
  FormattingOptions options;

  static llvm::Optional<DocumentFormattingParams>
  parse(llvm::yaml::MappingNode *Params);
};

} // namespace clangd
} // namespace clang

#endif
