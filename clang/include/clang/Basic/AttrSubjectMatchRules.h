//===-- AttrSubjectMatchRules.h - Attribute subject match rules -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ATTR_SUBJECT_MATCH_RULES_H
#define LLVM_CLANG_BASIC_ATTR_SUBJECT_MATCH_RULES_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
namespace attr {

/// \brief A list of all the recognized kinds of attributes.
enum SubjectMatchRule {
#define ATTR_MATCH_RULE(X, Spelling, IsAbstract) X,
#include "clang/Basic/AttrSubMatchRulesList.inc"
};

const char *getSubjectMatchRuleSpelling(SubjectMatchRule Rule);

using ParsedSubjectMatchRuleSet = llvm::DenseMap<SubjectMatchRule, SourceRange>;

} // end namespace attr
} // end namespace clang

namespace llvm {

template <>
struct DenseMapInfo<clang::attr::SubjectMatchRule> : DenseMapInfo<int> {
  static inline clang::attr::SubjectMatchRule getEmptyKey() {
    return (clang::attr::SubjectMatchRule)DenseMapInfo<int>::getEmptyKey();
  }
  static inline clang::attr::SubjectMatchRule getTombstoneKey() {
    return (clang::attr::SubjectMatchRule)DenseMapInfo<int>::getTombstoneKey();
  }
};

} // end namespace llvm

#endif
