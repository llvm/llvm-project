//===--- Builtins.cpp - Builtin function implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements various things for builtin functions.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;

static const Builtin::Info BuiltinInfo[] = {
  { "not a builtin function", 0, 0, 0, ALL_LANGUAGES },
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, ALL_LANGUAGES },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER, BUILTIN_LANG) { #ID, TYPE, ATTRS, HEADER,\
                                                            BUILTIN_LANG },
#include "clang/Basic/Builtins.def"
};

const Builtin::Info &Builtin::Context::GetRecord(unsigned ID) const {
  if (ID < Builtin::FirstTSBuiltin)
    return BuiltinInfo[ID];
  assert(ID - Builtin::FirstTSBuiltin < NumTSRecords && "Invalid builtin ID!");
  return TSRecords[ID - Builtin::FirstTSBuiltin];
}

Builtin::Context::Context() {
  // Get the target specific builtins from the target.
  TSRecords = 0;
  NumTSRecords = 0;
}

void Builtin::Context::InitializeTarget(const TargetInfo &Target) {
  assert(NumTSRecords == 0 && "Already initialized target?");
  Target.getTargetBuiltins(TSRecords, NumTSRecords);  
}

/// InitializeBuiltins - Mark the identifiers for all the builtins with their
/// appropriate builtin ID # and mark any non-portable builtin identifiers as
/// such.
void Builtin::Context::InitializeBuiltins(IdentifierTable &Table,
                                          const LangOptions& LangOpts) {
  // Step #1: mark all target-independent builtins with their ID's.
  for (unsigned i = Builtin::NotBuiltin+1; i != Builtin::FirstTSBuiltin; ++i)
    if (!LangOpts.NoBuiltin || !strchr(BuiltinInfo[i].Attributes, 'f')) {
      if (LangOpts.ObjC1 || 
          BuiltinInfo[i].builtin_lang != clang::OBJC_LANG)
        Table.get(BuiltinInfo[i].Name).setBuiltinID(i);
    }

  // Step #2: Register target-specific builtins.
  for (unsigned i = 0, e = NumTSRecords; i != e; ++i)
    if (!LangOpts.NoBuiltin || !strchr(TSRecords[i].Attributes, 'f'))
      Table.get(TSRecords[i].Name).setBuiltinID(i+Builtin::FirstTSBuiltin);
}

void
Builtin::Context::GetBuiltinNames(SmallVectorImpl<const char *> &Names,
                                  bool NoBuiltins) {
  // Final all target-independent names
  for (unsigned i = Builtin::NotBuiltin+1; i != Builtin::FirstTSBuiltin; ++i)
    if (!NoBuiltins || !strchr(BuiltinInfo[i].Attributes, 'f'))
      Names.push_back(BuiltinInfo[i].Name);

  // Find target-specific names.
  for (unsigned i = 0, e = NumTSRecords; i != e; ++i)
    if (!NoBuiltins || !strchr(TSRecords[i].Attributes, 'f'))
      Names.push_back(TSRecords[i].Name);
}

void Builtin::Context::ForgetBuiltin(unsigned ID, IdentifierTable &Table) {
  Table.get(GetRecord(ID).Name).setBuiltinID(0);
}

bool
Builtin::Context::isPrintfLike(unsigned ID, unsigned &FormatIdx,
                               bool &HasVAListArg) {
  const char *Printf = strpbrk(GetRecord(ID).Attributes, "pP");
  if (!Printf)
    return false;

  HasVAListArg = (*Printf == 'P');

  ++Printf;
  assert(*Printf == ':' && "p or P specifier must have be followed by a ':'");
  ++Printf;

  assert(strchr(Printf, ':') && "printf specifier must end with a ':'");
  FormatIdx = strtol(Printf, 0, 10);
  return true;
}

// FIXME: Refactor with isPrintfLike.
bool
Builtin::Context::isScanfLike(unsigned ID, unsigned &FormatIdx,
                              bool &HasVAListArg) {
  const char *Scanf = strpbrk(GetRecord(ID).Attributes, "sS");
  if (!Scanf)
    return false;

  HasVAListArg = (*Scanf == 'S');

  ++Scanf;
  assert(*Scanf == ':' && "s or S specifier must have be followed by a ':'");
  ++Scanf;

  assert(strchr(Scanf, ':') && "printf specifier must end with a ':'");
  FormatIdx = strtol(Scanf, 0, 10);
  return true;
}

