//===-- TweakTesting.cpp ------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"

#include "SourceCode.h"
#include "TestTU.h"
#include "refactor/Tweak.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

namespace clang {
namespace clangd {
namespace {
using Context = TweakTest::CodeContext;

std::pair<llvm::StringRef, llvm::StringRef> wrapping(Context Ctx) {
  switch (Ctx) {
  case TweakTest::File:
    return {"", ""};
  case TweakTest::Function:
    return {"void wrapperFunction(){\n", "\n}"};
  case TweakTest::Expression:
    return {"auto expressionWrapper(){return\n", "\n;}"};
  }
  llvm_unreachable("Unknown TweakTest::CodeContext enum");
}

std::string wrap(Context Ctx, llvm::StringRef Inner) {
  auto Wrapping = wrapping(Ctx);
  return (Wrapping.first + Inner + Wrapping.second).str();
}

llvm::StringRef unwrap(Context Ctx, llvm::StringRef Outer) {
  auto Wrapping = wrapping(Ctx);
  // Unwrap only if the code matches the expected wrapping.
  // Don't allow the begin/end wrapping to overlap!
  if (Outer.startswith(Wrapping.first) && Outer.endswith(Wrapping.second) &&
      Outer.size() >= Wrapping.first.size() + Wrapping.second.size())
    return Outer.drop_front(Wrapping.first.size())
        .drop_back(Wrapping.second.size());
  return Outer;
}

llvm::Annotations::Range rangeOrPoint(const llvm::Annotations &A) {
  if (A.points().size() != 0) {
    assert(A.ranges().size() == 0 &&
           "both a cursor point and a selection range were specified");
    return {A.point(), A.point()};
  }
  return A.range();
}

// Prepare and apply the specified tweak based on the selection in Input.
// Returns None if and only if prepare() failed.
llvm::Optional<llvm::Expected<Tweak::Effect>>
applyTweak(ParsedAST &AST, llvm::Annotations::Range Range, StringRef TweakID,
           const SymbolIndex *Index, llvm::vfs::FileSystem *FS) {
  llvm::Optional<llvm::Expected<Tweak::Effect>> Result;
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(), Range.Begin,
                            Range.End, [&](SelectionTree ST) {
                              Tweak::Selection S(Index, AST, Range.Begin,
                                                 Range.End, std::move(ST), FS);
                              if (auto T = prepareTweak(TweakID, S, nullptr)) {
                                Result = (*T)->apply(S);
                                return true;
                              } else {
                                llvm::consumeError(T.takeError());
                                return false;
                              }
                            });
  return Result;
}

} // namespace

std::string TweakTest::apply(llvm::StringRef MarkedCode,
                             llvm::StringMap<std::string> *EditedFiles) const {
  std::string WrappedCode = wrap(Context, MarkedCode);
  llvm::Annotations Input(WrappedCode);
  TestTU TU;
  TU.Filename = std::string(FileName);
  TU.HeaderCode = Header;
  TU.AdditionalFiles = std::move(ExtraFiles);
  TU.Code = std::string(Input.code());
  TU.ExtraArgs = ExtraArgs;
  ParsedAST AST = TU.build();

  auto Result = applyTweak(
      AST, rangeOrPoint(Input), TweakID, Index.get(),
      &AST.getSourceManager().getFileManager().getVirtualFileSystem());
  if (!Result)
    return "unavailable";
  if (!*Result)
    return "fail: " + llvm::toString(Result->takeError());
  const auto &Effect = **Result;
  if ((*Result)->ShowMessage)
    return "message:\n" + *Effect.ShowMessage;
  if (Effect.ApplyEdits.empty())
    return "no effect";

  std::string EditedMainFile;
  for (auto &It : Effect.ApplyEdits) {
    auto NewText = It.second.apply();
    if (!NewText)
      return "bad edits: " + llvm::toString(NewText.takeError());
    llvm::StringRef Unwrapped = unwrap(Context, *NewText);
    if (It.first() == testPath(TU.Filename))
      EditedMainFile = std::string(Unwrapped);
    else {
      if (!EditedFiles)
        ADD_FAILURE() << "There were changes to additional files, but client "
                         "provided a nullptr for EditedFiles.";
      else
        EditedFiles->insert_or_assign(It.first(), Unwrapped.str());
    }
  }
  return EditedMainFile;
}

bool TweakTest::isAvailable(WrappedAST &AST,
                            llvm::Annotations::Range Range) const {
  // Adjust range for wrapping offset.
  Range.Begin += AST.second;
  Range.End += AST.second;
  auto Result = applyTweak(
      AST.first, Range, TweakID, Index.get(),
      &AST.first.getSourceManager().getFileManager().getVirtualFileSystem());
  // We only care if prepare() succeeded, but must handle Errors.
  if (Result && !*Result)
    consumeError(Result->takeError());
  return Result.hasValue();
}

TweakTest::WrappedAST TweakTest::build(llvm::StringRef Code) const {
  TestTU TU;
  TU.Filename = std::string(FileName);
  TU.HeaderCode = Header;
  TU.Code = wrap(Context, Code);
  TU.ExtraArgs = ExtraArgs;
  TU.AdditionalFiles = std::move(ExtraFiles);
  return {TU.build(), wrapping(Context).first.size()};
}

std::string TweakTest::decorate(llvm::StringRef Code, unsigned Point) {
  return (Code.substr(0, Point) + "^" + Code.substr(Point)).str();
}

std::string TweakTest::decorate(llvm::StringRef Code,
                                llvm::Annotations::Range Range) {
  return (Code.substr(0, Range.Begin) + "[[" +
          Code.substr(Range.Begin, Range.End) + "]]" + Code.substr(Range.End))
      .str();
}

} // namespace clangd
} // namespace clang
