//===---  BugType.h - Bug Information Description ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines BugType, a class representing a bug type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_BUGTYPE_H
#define LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_BUGTYPE_H

#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include <string>
#include <variant>

namespace clang {

namespace ento {

class BugReporter;

class BugType {
private:
  struct CheckerPartRef {
    const CheckerBase *Checker;
    CheckerPartIdx Idx;
  };
  using CheckerNameInfo = std::variant<CheckerNameRef, CheckerPartRef>;

  const CheckerNameInfo CheckerName;
  const std::string Description;
  const std::string Category;
  bool SuppressOnSink;

  virtual void anchor();

public:
  // Straightforward constructor where the checker name is specified directly.
  // TODO: As far as I know all applications of this constructor involve ugly
  // hacks that could be avoided by switching to a different constructor.
  // When those are all eliminated, this constructor should be removed to
  // eliminate the `variant` and simplify this class.
  BugType(CheckerNameRef CheckerName, StringRef Desc,
          StringRef Cat = categories::LogicError, bool SuppressOnSink = false)
      : CheckerName(CheckerName), Description(Desc), Category(Cat),
        SuppressOnSink(SuppressOnSink) {}
  // Constructor that can be called from the constructor of a checker object.
  // At that point the checker name is not yet available, but we can save a
  // pointer to the checker and later use that to query the name.
  BugType(const CheckerBase *Checker, StringRef Desc,
          StringRef Cat = categories::LogicError, bool SuppressOnSink = false)
      : CheckerName(CheckerPartRef{Checker, DefaultPart}), Description(Desc),
        Category(Cat), SuppressOnSink(SuppressOnSink) {}
  // Constructor that can be called from the constructor of a checker object
  // when it has multiple parts with separate names. We save the name and the
  // part index to be able to query the name of that part later.
  BugType(const CheckerBase *Checker, CheckerPartIdx Idx, StringRef Desc,
          StringRef Cat = categories::LogicError, bool SuppressOnSink = false)
      : CheckerName(CheckerPartRef{Checker, Idx}), Description(Desc),
        Category(Cat), SuppressOnSink(SuppressOnSink) {}
  virtual ~BugType() = default;

  StringRef getDescription() const { return Description; }
  StringRef getCategory() const { return Category; }
  StringRef getCheckerName() const {
    if (const auto *CNR = std::get_if<CheckerNameRef>(&CheckerName))
      return *CNR;

    auto [Checker, Idx] = std::get<CheckerPartRef>(CheckerName);
    return Checker->getName(Idx);
  }

  /// isSuppressOnSink - Returns true if bug reports associated with this bug
  ///  type should be suppressed if the end node of the report is post-dominated
  ///  by a sink node.
  bool isSuppressOnSink() const { return SuppressOnSink; }
};

} // namespace ento

} // end clang namespace
#endif
