//===-- include/flang/Parser/user-state.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_USER_STATE_H_
#define FORTRAN_PARSER_USER_STATE_H_

// Instances of ParseState (parse-state.h) incorporate instances of this
// UserState class, which encapsulates any semantic information necessary for
// parse tree construction so as to avoid any need for representing
// state in static data.

#include "flang/Common/Fortran-features.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/Support/raw_ostream.h"
#include <cinttypes>
#include <optional>
#include <set>
#include <unordered_map>

namespace Fortran::parser {

class AllCookedSources;
class ParsingLog;
class ParseState;

class Success {}; // for when one must return something that's present

class UserState {
public:
  UserState(const AllCookedSources &allCooked,
      common::LanguageFeatureControl features)
      : allCooked_{allCooked}, features_{features} {}

  const AllCookedSources &allCooked() const { return allCooked_; }
  const common::LanguageFeatureControl &features() const { return features_; }

  llvm::raw_ostream *debugOutput() const { return debugOutput_; }
  UserState &set_debugOutput(llvm::raw_ostream &out) {
    debugOutput_ = &out;
    return *this;
  }

  ParsingLog *log() const { return log_; }
  UserState &set_log(ParsingLog *log) {
    log_ = log;
    return *this;
  }

  bool instrumentedParse() const { return instrumentedParse_; }
  UserState &set_instrumentedParse(bool yes) {
    instrumentedParse_ = yes;
    return *this;
  }

  void NewSubprogram() {
    doLabels_.clear();
    nonlabelDoConstructNestingDepth_ = 0;
    oldStructureComponents_.clear();
  }

  using Label = std::uint64_t;
  bool IsDoLabel(Label label) const {
    auto iter{doLabels_.find(label)};
    return iter != doLabels_.end() &&
        iter->second >= nonlabelDoConstructNestingDepth_;
  }
  void NewDoLabel(Label label) {
    doLabels_[label] = nonlabelDoConstructNestingDepth_;
  }

  void EnterNonlabelDoConstruct() { ++nonlabelDoConstructNestingDepth_; }
  void LeaveDoConstruct() {
    if (nonlabelDoConstructNestingDepth_ > 0) {
      --nonlabelDoConstructNestingDepth_;
    }
  }

  void NoteOldStructureComponent(const CharBlock &name) {
    oldStructureComponents_.insert(name);
  }
  bool IsOldStructureComponent(const CharBlock &name) const {
    return oldStructureComponents_.find(name) != oldStructureComponents_.end();
  }

private:
  const AllCookedSources &allCooked_;

  llvm::raw_ostream *debugOutput_{nullptr};

  ParsingLog *log_{nullptr};
  bool instrumentedParse_{false};

  std::unordered_map<Label, int> doLabels_;
  int nonlabelDoConstructNestingDepth_{0};

  std::set<CharBlock> oldStructureComponents_;

  common::LanguageFeatureControl features_;
};

// Definitions of parser classes that manipulate the UserState.
struct StartNewSubprogram {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState &);
};

struct CapturedLabelDoStmt {
  using resultType = Statement<common::Indirection<LabelDoStmt>>;
  static std::optional<resultType> Parse(ParseState &);
};

struct EndDoStmtForCapturedLabelDoStmt {
  using resultType = Statement<common::Indirection<EndDoStmt>>;
  static std::optional<resultType> Parse(ParseState &);
};

struct EnterNonlabelDoConstruct {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState &);
};

struct LeaveDoConstruct {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState &);
};

struct OldStructureComponentName {
  using resultType = Name;
  static std::optional<Name> Parse(ParseState &);
};

struct StructureComponents {
  using resultType = DataComponentDefStmt;
  static std::optional<DataComponentDefStmt> Parse(ParseState &);
};
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_USER_STATE_H_
