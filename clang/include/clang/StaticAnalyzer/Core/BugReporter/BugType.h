//===---  BugType.h - Bug Information Desciption ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

namespace clang {

namespace ento {

class BugReporter;
class ExplodedNode;
class ExprEngine;

class BugType {
private:
  const CheckName Check;
  const std::string Name;
  const std::string Category;
  bool SuppressonSink;

  virtual void anchor();
public:
  BugType(class CheckName check, StringRef name, StringRef cat)
      : Check(check), Name(name), Category(cat), SuppressonSink(false) {}
  BugType(const CheckerBase *checker, StringRef name, StringRef cat)
      : Check(checker->getCheckName()), Name(name), Category(cat),
        SuppressonSink(false) {}
  virtual ~BugType() {}

  // FIXME: Should these be made strings as well?
  StringRef getName() const { return Name; }
  StringRef getCategory() const { return Category; }
  StringRef getCheckName() const { return Check.getName(); }

  /// isSuppressOnSink - Returns true if bug reports associated with this bug
  ///  type should be suppressed if the end node of the report is post-dominated
  ///  by a sink node.
  bool isSuppressOnSink() const { return SuppressonSink; }
  void setSuppressOnSink(bool x) { SuppressonSink = x; }

  virtual void FlushReports(BugReporter& BR);
};

class BuiltinBug : public BugType {
  const std::string desc;
  void anchor() override;
public:
  BuiltinBug(class CheckName check, const char *name, const char *description)
      : BugType(check, name, categories::LogicError), desc(description) {}

  BuiltinBug(const CheckerBase *checker, const char *name,
             const char *description)
      : BugType(checker, name, categories::LogicError), desc(description) {}

  BuiltinBug(const CheckerBase *checker, const char *name)
      : BugType(checker, name, categories::LogicError), desc(name) {}

  StringRef getDescription() const { return desc; }
};

} // end GR namespace

} // end clang namespace
#endif
