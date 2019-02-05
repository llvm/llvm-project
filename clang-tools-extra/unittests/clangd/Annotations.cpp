//===--- Annotations.cpp - Annotated source code for unit tests --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "SourceCode.h"

namespace clang {
namespace clangd {

// Crash if the assertion fails, printing the message and testcase.
// More elegant error handling isn't needed for unit tests.
static void require(bool Assertion, const char *Msg, llvm::StringRef Code) {
  if (!Assertion) {
    llvm::errs() << "Annotated testcase: " << Msg << "\n" << Code << "\n";
    llvm_unreachable("Annotated testcase assertion failed!");
  }
}

Annotations::Annotations(llvm::StringRef Text) {
  auto Here = [this] { return offsetToPosition(Code, Code.size()); };
  auto Require = [Text](bool Assertion, const char *Msg) {
    require(Assertion, Msg, Text);
  };
  llvm::Optional<llvm::StringRef> Name;
  llvm::SmallVector<std::pair<llvm::StringRef, Position>, 8> OpenRanges;

  Code.reserve(Text.size());
  while (!Text.empty()) {
    if (Text.consume_front("^")) {
      Points[Name.getValueOr("")].push_back(Here());
      Name = None;
      continue;
    }
    if (Text.consume_front("[[")) {
      OpenRanges.emplace_back(Name.getValueOr(""), Here());
      Name = None;
      continue;
    }
    Require(!Name, "$name should be followed by ^ or [[");
    if (Text.consume_front("]]")) {
      Require(!OpenRanges.empty(), "unmatched ]]");
      Ranges[OpenRanges.back().first].push_back(
          {OpenRanges.back().second, Here()});
      OpenRanges.pop_back();
      continue;
    }
    if (Text.consume_front("$")) {
      Name = Text.take_while(llvm::isAlnum);
      Text = Text.drop_front(Name->size());
      continue;
    }
    Code.push_back(Text.front());
    Text = Text.drop_front();
  }
  Require(!Name, "unterminated $name");
  Require(OpenRanges.empty(), "unmatched [[");
}

Position Annotations::point(llvm::StringRef Name) const {
  auto I = Points.find(Name);
  require(I != Points.end() && I->getValue().size() == 1,
          "expected exactly one point", Code);
  return I->getValue()[0];
}
std::vector<Position> Annotations::points(llvm::StringRef Name) const {
  auto P = Points.lookup(Name);
  return {P.begin(), P.end()};
}
Range Annotations::range(llvm::StringRef Name) const {
  auto I = Ranges.find(Name);
  require(I != Ranges.end() && I->getValue().size() == 1,
          "expected exactly one range", Code);
  return I->getValue()[0];
}
std::vector<Range> Annotations::ranges(llvm::StringRef Name) const {
  auto R = Ranges.lookup(Name);
  return {R.begin(), R.end()};
}

} // namespace clangd
} // namespace clang
