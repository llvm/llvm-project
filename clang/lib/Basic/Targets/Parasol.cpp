//===--- Parasol.cpp - Implement Parasol target feature support ---------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file implements ParasolTargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "Parasol.h"
#include "clang/Basic/MacroBuilder.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

const char *const ParasolTargetInfo::GCCRegNames[] = {
  // Integer registers
  "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
  "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
  "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
  "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31",
};

const TargetInfo::GCCRegAlias GCCRegAliases[] = {
  {{"zero"}, "x0"}, {{"ra"}, "x1"},   {{"sp"}, "x2"},    {{"gp"}, "x3"},
  {{"tp"}, "x4"},   {{"t0"}, "x5"},   {{"t1"}, "x6"},    {{"t2"}, "x7"},
  {{"s0"}, "x8"},   {{"s1"}, "x9"},   {{"a0"}, "x10"},   {{"a1"}, "x11"},
  {{"a2"}, "x12"},  {{"a3"}, "x13"},  {{"a4"}, "x14"},   {{"a5"}, "x15"},
  {{"a6"}, "x16"},  {{"a7"}, "x17"},  {{"s2"}, "x18"},   {{"s3"}, "x19"},
  {{"s4"}, "x20"},  {{"s5"}, "x21"},  {{"s6"}, "x22"},   {{"s7"}, "x23"},
  {{"s8"}, "x24"},  {{"s9"}, "x25"},  {{"s10"}, "x26"},  {{"s11"}, "x27"},
  {{"t3"}, "x28"},  {{"t4"}, "x29"},  {{"t5"}, "x30"},   {{"t6"}, "x31"}};

ArrayRef<const char *> ParasolTargetInfo::getGCCRegNames() const {
  return llvm::ArrayRef(GCCRegNames);
}

ArrayRef<TargetInfo::GCCRegAlias> ParasolTargetInfo::getGCCRegAliases() const {
  return llvm::ArrayRef(GCCRegAliases);
}

void ParasolTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  // Define the __Parasol__ macro when building for this target
  Builder.defineMacro("__Parasol__");
}
