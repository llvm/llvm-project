//===--- ProtocolHandlers.cpp - LSP callbacks -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProtocolHandlers.h"
#include "ASTManager.h"
#include "DocumentStore.h"
#include "clang/Format/Format.h"
using namespace clang;
using namespace clangd;

void TextDocumentDidOpenHandler::handleNotification(
    llvm::yaml::MappingNode *Params) {
  auto DOTDP = DidOpenTextDocumentParams::parse(Params);
  if (!DOTDP) {
    Output.log("Failed to decode DidOpenTextDocumentParams!\n");
    return;
  }
  Store.addDocument(DOTDP->textDocument.uri.file, DOTDP->textDocument.text);
}

void TextDocumentDidCloseHandler::handleNotification(
    llvm::yaml::MappingNode *Params) {
  auto DCTDP = DidCloseTextDocumentParams::parse(Params);
  if (!DCTDP) {
    Output.log("Failed to decode DidCloseTextDocumentParams!\n");
    return;
  }

  Store.removeDocument(DCTDP->textDocument.uri.file);
}

void TextDocumentDidChangeHandler::handleNotification(
    llvm::yaml::MappingNode *Params) {
  auto DCTDP = DidChangeTextDocumentParams::parse(Params);
  if (!DCTDP || DCTDP->contentChanges.size() != 1) {
    Output.log("Failed to decode DidChangeTextDocumentParams!\n");
    return;
  }
  // We only support full syncing right now.
  Store.addDocument(DCTDP->textDocument.uri.file, DCTDP->contentChanges[0].text);
}

/// Turn a [line, column] pair into an offset in Code.
static size_t positionToOffset(StringRef Code, Position P) {
  size_t Offset = 0;
  for (int I = 0; I != P.line; ++I) {
    // FIXME: \r\n
    // FIXME: UTF-8
    size_t F = Code.find('\n', Offset);
    if (F == StringRef::npos)
      return 0; // FIXME: Is this reasonable?
    Offset = F + 1;
  }
  return (Offset == 0 ? 0 : (Offset - 1)) + P.character;
}

/// Turn an offset in Code into a [line, column] pair.
static Position offsetToPosition(StringRef Code, size_t Offset) {
  StringRef JustBefore = Code.substr(0, Offset);
  // FIXME: \r\n
  // FIXME: UTF-8
  int Lines = JustBefore.count('\n');
  int Cols = JustBefore.size() - JustBefore.rfind('\n') - 1;
  return {Lines, Cols};
}

template <typename T>
static std::string replacementsToEdits(StringRef Code, const T &Replacements) {
  // Turn the replacements into the format specified by the Language Server
  // Protocol. Fuse them into one big JSON array.
  std::string Edits;
  for (auto &R : Replacements) {
    Range ReplacementRange = {
        offsetToPosition(Code, R.getOffset()),
        offsetToPosition(Code, R.getOffset() + R.getLength())};
    TextEdit TE = {ReplacementRange, R.getReplacementText()};
    Edits += TextEdit::unparse(TE);
    Edits += ',';
  }
  if (!Edits.empty())
    Edits.pop_back();

  return Edits;
}

static std::string formatCode(StringRef Code, StringRef Filename,
                              ArrayRef<tooling::Range> Ranges, StringRef ID) {
  // Call clang-format.
  // FIXME: Don't ignore style.
  format::FormatStyle Style = format::getLLVMStyle();
  tooling::Replacements Replacements =
      format::reformat(Style, Code, Ranges, Filename);

  std::string Edits = replacementsToEdits(Code, Replacements);
  return R"({"jsonrpc":"2.0","id":)" + ID.str() +
         R"(,"result":[)" + Edits + R"(]})";
}

void TextDocumentRangeFormattingHandler::handleMethod(
    llvm::yaml::MappingNode *Params, StringRef ID) {
  auto DRFP = DocumentRangeFormattingParams::parse(Params);
  if (!DRFP) {
    Output.log("Failed to decode DocumentRangeFormattingParams!\n");
    return;
  }

  std::string Code = Store.getDocument(DRFP->textDocument.uri.file);

  size_t Begin = positionToOffset(Code, DRFP->range.start);
  size_t Len = positionToOffset(Code, DRFP->range.end) - Begin;

  writeMessage(formatCode(Code, DRFP->textDocument.uri.file,
                          {clang::tooling::Range(Begin, Len)}, ID));
}

void TextDocumentOnTypeFormattingHandler::handleMethod(
    llvm::yaml::MappingNode *Params, StringRef ID) {
  auto DOTFP = DocumentOnTypeFormattingParams::parse(Params);
  if (!DOTFP) {
    Output.log("Failed to decode DocumentOnTypeFormattingParams!\n");
    return;
  }

  // Look for the previous opening brace from the character position and format
  // starting from there.
  std::string Code = Store.getDocument(DOTFP->textDocument.uri.file);
  size_t CursorPos = positionToOffset(Code, DOTFP->position);
  size_t PreviousLBracePos = StringRef(Code).find_last_of('{', CursorPos);
  if (PreviousLBracePos == StringRef::npos)
    PreviousLBracePos = CursorPos;
  size_t Len = 1 + CursorPos - PreviousLBracePos;

  writeMessage(formatCode(Code, DOTFP->textDocument.uri.file,
                          {clang::tooling::Range(PreviousLBracePos, Len)}, ID));
}

void TextDocumentFormattingHandler::handleMethod(
    llvm::yaml::MappingNode *Params, StringRef ID) {
  auto DFP = DocumentFormattingParams::parse(Params);
  if (!DFP) {
    Output.log("Failed to decode DocumentFormattingParams!\n");
    return;
  }

  // Format everything.
  std::string Code = Store.getDocument(DFP->textDocument.uri.file);
  writeMessage(formatCode(Code, DFP->textDocument.uri.file,
                          {clang::tooling::Range(0, Code.size())}, ID));
}

void CodeActionHandler::handleMethod(llvm::yaml::MappingNode *Params,
                                     StringRef ID) {
  auto CAP = CodeActionParams::parse(Params);
  if (!CAP) {
    Output.log("Failed to decode CodeActionParams!\n");
    return;
  }

  // We provide a code action for each diagnostic at the requested location
  // which has FixIts available.
  std::string Code = AST.getStore().getDocument(CAP->textDocument.uri.file);
  std::string Commands;
  for (Diagnostic &D : CAP->context.diagnostics) {
    std::vector<clang::tooling::Replacement> Fixes = AST.getFixIts(CAP->textDocument.uri.file, D);
    std::string Edits = replacementsToEdits(Code, Fixes);

    if (!Edits.empty())
      Commands +=
          R"({"title":"Apply FixIt ')" + llvm::yaml::escape(D.message) +
          R"('", "command": "clangd.applyFix", "arguments": [")" +
          llvm::yaml::escape(CAP->textDocument.uri.uri) +
          R"(", [)" + Edits +
          R"(]]},)";
  }
  if (!Commands.empty())
    Commands.pop_back();

  writeMessage(
      R"({"jsonrpc":"2.0","id":)" + ID.str() +
      R"(, "result": [)" + Commands +
      R"(]})");
}

void CompletionHandler::handleMethod(llvm::yaml::MappingNode *Params,
                                     StringRef ID) {
  auto TDPP = TextDocumentPositionParams::parse(Params);
  if (!TDPP) {
    Output.log("Failed to decode TextDocumentPositionParams!\n");
    return;
  }

  auto Items = AST.codeComplete(TDPP->textDocument.uri.file, TDPP->position.line,
                                TDPP->position.character);
  std::string Completions;
  for (const auto &Item : Items) {
    Completions += CompletionItem::unparse(Item);
    Completions += ",";
  }
  if (!Completions.empty())
    Completions.pop_back();
  writeMessage(
      R"({"jsonrpc":"2.0","id":)" + ID.str() +
      R"(,"result":[)" + Completions + R"(]})");
}
