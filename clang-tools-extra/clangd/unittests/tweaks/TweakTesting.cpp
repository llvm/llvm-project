//===-- TweakTesting.cpp ------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"

#include "Annotations.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "refactor/Tweak.h"
#include "clang/Tooling/Core/Replacement.h"
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

std::pair<unsigned, unsigned> rangeOrPoint(const Annotations &A) {
  Range SelectionRng;
  if (A.points().size() != 0) {
    assert(A.ranges().size() == 0 &&
           "both a cursor point and a selection range were specified");
    SelectionRng = Range{A.point(), A.point()};
  } else {
    SelectionRng = A.range();
  }
  return {cantFail(positionToOffset(A.code(), SelectionRng.start)),
          cantFail(positionToOffset(A.code(), SelectionRng.end))};
}

// Prepare and apply the specified tweak based on the selection in Input.
// Returns None if and only if prepare() failed.
llvm::Optional<llvm::Expected<Tweak::Effect>>
applyTweak(ParsedAST &AST, const Annotations &Input, StringRef TweakID,
           const SymbolIndex *Index) {
  auto Range = rangeOrPoint(Input);
  llvm::Optional<llvm::Expected<Tweak::Effect>> Result;
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(), Range.first,
                            Range.second, [&](SelectionTree ST) {
                              Tweak::Selection S(Index, AST, Range.first,
                                                 Range.second, std::move(ST));
                              if (auto T = prepareTweak(TweakID, S)) {
                                Result = (*T)->apply(S);
                                return true;
                              } else {
                                llvm::consumeError(T.takeError());
                                return false;
                              }
                            });
  return Result;
}

MATCHER_P7(TweakIsAvailable, TweakID, Ctx, Header, ExtraArgs, ExtraFiles, Index,
           FileName,
           (TweakID + (negation ? " is unavailable" : " is available")).str()) {
  std::string WrappedCode = wrap(Ctx, arg);
  Annotations Input(WrappedCode);
  TestTU TU;
  TU.Filename = std::string(FileName);
  TU.HeaderCode = Header;
  TU.Code = std::string(Input.code());
  TU.ExtraArgs = ExtraArgs;
  TU.AdditionalFiles = std::move(ExtraFiles);
  ParsedAST AST = TU.build();
  auto Result = applyTweak(AST, Input, TweakID, Index);
  // We only care if prepare() succeeded, but must handle Errors.
  if (Result && !*Result)
    consumeError(Result->takeError());
  return Result.hasValue();
}

} // namespace

std::string TweakTest::apply(llvm::StringRef MarkedCode,
                             llvm::StringMap<std::string> *EditedFiles) const {
  std::string WrappedCode = wrap(Context, MarkedCode);
  Annotations Input(WrappedCode);
  TestTU TU;
  TU.Filename = std::string(FileName);
  TU.HeaderCode = Header;
  TU.AdditionalFiles = std::move(ExtraFiles);
  TU.Code = std::string(Input.code());
  TU.ExtraArgs = ExtraArgs;
  ParsedAST AST = TU.build();

  auto Result = applyTweak(AST, Input, TweakID, Index.get());
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

::testing::Matcher<llvm::StringRef> TweakTest::isAvailable() const {
  return TweakIsAvailable(llvm::StringRef(TweakID), Context, Header, ExtraArgs,
                          ExtraFiles, Index.get(), FileName);
}

std::vector<std::string> TweakTest::expandCases(llvm::StringRef MarkedCode) {
  Annotations Test(MarkedCode);
  llvm::StringRef Code = Test.code();
  std::vector<std::string> Cases;
  for (const auto &Point : Test.points()) {
    size_t Offset = llvm::cantFail(positionToOffset(Code, Point));
    Cases.push_back((Code.substr(0, Offset) + "^" + Code.substr(Offset)).str());
  }
  for (const auto &Range : Test.ranges()) {
    size_t Begin = llvm::cantFail(positionToOffset(Code, Range.start));
    size_t End = llvm::cantFail(positionToOffset(Code, Range.end));
    Cases.push_back((Code.substr(0, Begin) + "[[" +
                     Code.substr(Begin, End - Begin) + "]]" + Code.substr(End))
                        .str());
  }
  assert(!Cases.empty() && "No markings in MarkedCode?");
  return Cases;
}

} // namespace clangd
} // namespace clang
