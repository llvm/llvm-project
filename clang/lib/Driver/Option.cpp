//===--- Option.cpp - Abstract Driver Options -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Option.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <algorithm>
using namespace clang::driver;

Option::Option(const OptTable::Info *info, const OptTable *owner)
  : Info(info), Owner(owner) {

  // Multi-level aliases are not supported, and alias options cannot
  // have groups. This just simplifies option tracking, it is not an
  // inherent limitation.
  assert((!Info || !getAlias().isValid() || (!getAlias().getAlias().isValid() &&
         !getGroup().isValid())) &&
         "Multi-level aliases and aliases with groups are unsupported.");
}

Option::~Option() {
}

void Option::dump() const {
  llvm::errs() << "<";
  switch (getKind()) {
#define P(N) case N: llvm::errs() << #N; break
    P(GroupClass);
    P(InputClass);
    P(UnknownClass);
    P(FlagClass);
    P(JoinedClass);
    P(SeparateClass);
    P(CommaJoinedClass);
    P(MultiArgClass);
    P(JoinedOrSeparateClass);
    P(JoinedAndSeparateClass);
#undef P
  }

  llvm::errs() << " Prefixes:[";
  for (const char * const *Pre = Info->Prefixes; *Pre != 0; ++Pre) {
    llvm::errs() << '"' << *Pre << (*(Pre + 1) == 0 ? "\"" : "\", ");
  }
  llvm::errs() << ']';

  llvm::errs() << " Name:\"" << getName() << '"';

  const Option Group = getGroup();
  if (Group.isValid()) {
    llvm::errs() << " Group:";
    Group.dump();
  }

  const Option Alias = getAlias();
  if (Alias.isValid()) {
    llvm::errs() << " Alias:";
    Alias.dump();
  }

  if (getKind() == MultiArgClass)
    llvm::errs() << " NumArgs:" << getNumArgs();

  llvm::errs() << ">\n";
}

bool Option::matches(OptSpecifier Opt) const {
  // Aliases are never considered in matching, look through them.
  const Option Alias = getAlias();
  if (Alias.isValid())
    return Alias.matches(Opt);

  // Check exact match.
  if (getID() == Opt.getID())
    return true;

  const Option Group = getGroup();
  if (Group.isValid())
    return Group.matches(Opt);
  return false;
}

Arg *Option::accept(const ArgList &Args,
                    unsigned &Index,
                    unsigned ArgSize) const {
  const Option &UnaliasedOption = getUnaliasedOption();
  StringRef Spelling;
  // If the option was an alias, get the spelling from the unaliased one.
  if (getID() == UnaliasedOption.getID()) {
    Spelling = StringRef(Args.getArgString(Index), ArgSize);
  } else {
    Spelling = Args.MakeArgString(Twine(UnaliasedOption.getPrefix()) +
                                  Twine(UnaliasedOption.getName()));
  }

  switch (getKind()) {
  case FlagClass:
    if (ArgSize != strlen(Args.getArgString(Index)))
      return 0;

    return new Arg(UnaliasedOption, Spelling, Index++);
  case JoinedClass: {
    const char *Value = Args.getArgString(Index) + ArgSize;
    return new Arg(UnaliasedOption, Spelling, Index++, Value);
  }
  case CommaJoinedClass: {
    // Always matches.
    const char *Str = Args.getArgString(Index) + ArgSize;
    Arg *A = new Arg(UnaliasedOption, Spelling, Index++);

    // Parse out the comma separated values.
    const char *Prev = Str;
    for (;; ++Str) {
      char c = *Str;

      if (!c || c == ',') {
        if (Prev != Str) {
          char *Value = new char[Str - Prev + 1];
          memcpy(Value, Prev, Str - Prev);
          Value[Str - Prev] = '\0';
          A->getValues().push_back(Value);
        }

        if (!c)
          break;

        Prev = Str + 1;
      }
    }
    A->setOwnsValues(true);

    return A;
  }
  case SeparateClass:
    // Matches iff this is an exact match.
    // FIXME: Avoid strlen.
    if (ArgSize != strlen(Args.getArgString(Index)))
      return 0;

    Index += 2;
    if (Index > Args.getNumInputArgStrings())
      return 0;

    return new Arg(UnaliasedOption, Spelling,
                   Index - 2, Args.getArgString(Index - 1));
  case MultiArgClass: {
    // Matches iff this is an exact match.
    // FIXME: Avoid strlen.
    if (ArgSize != strlen(Args.getArgString(Index)))
      return 0;

    Index += 1 + getNumArgs();
    if (Index > Args.getNumInputArgStrings())
      return 0;

    Arg *A = new Arg(UnaliasedOption, Spelling, Index - 1 - getNumArgs(),
                      Args.getArgString(Index - getNumArgs()));
    for (unsigned i = 1; i != getNumArgs(); ++i)
      A->getValues().push_back(Args.getArgString(Index - getNumArgs() + i));
    return A;
  }
  case JoinedOrSeparateClass: {
    // If this is not an exact match, it is a joined arg.
    // FIXME: Avoid strlen.
    if (ArgSize != strlen(Args.getArgString(Index))) {
      const char *Value = Args.getArgString(Index) + ArgSize;
      return new Arg(*this, Spelling, Index++, Value);
    }

    // Otherwise it must be separate.
    Index += 2;
    if (Index > Args.getNumInputArgStrings())
      return 0;

    return new Arg(UnaliasedOption, Spelling,
                   Index - 2, Args.getArgString(Index - 1));
  }
  case JoinedAndSeparateClass:
    // Always matches.
    Index += 2;
    if (Index > Args.getNumInputArgStrings())
      return 0;

    return new Arg(UnaliasedOption, Spelling, Index - 2,
                   Args.getArgString(Index - 2) + ArgSize,
                   Args.getArgString(Index - 1));
  default:
    llvm_unreachable("Invalid option kind!");
  }
}
