//===--- AArch64.cpp - Implement AArch64 target feature support -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements AArch64 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/ArrayRef.h"

using namespace clang;
using namespace clang::targets;

const char *const AArch64TargetInfo::GCCRegNames[] = {
    // 32-bit Integer registers
    "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11",
    "w12", "w13", "w14", "w15", "w16", "w17", "w18", "w19", "w20", "w21", "w22",
    "w23", "w24", "w25", "w26", "w27", "w28", "w29", "w30", "wsp",

    // 64-bit Integer registers
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
    "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22",
    "x23", "x24", "x25", "x26", "x27", "x28", "fp", "lr", "sp",

    // 32-bit floating point regsisters
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22",
    "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",

    // 64-bit floating point regsisters
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
    "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
    "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",

    // Vector registers
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
    "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
    "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
};

ArrayRef<const char *> AArch64TargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

const TargetInfo::GCCRegAlias AArch64TargetInfo::GCCRegAliases[] = {
    {{"w31"}, "wsp"}, {{"x29"}, "fp"}, {{"x30"}, "lr"}, {{"x31"}, "sp"},
    // The S/D/Q and W/X registers overlap, but aren't really aliases; we
    // don't want to substitute one of these for a different-sized one.
};

ArrayRef<TargetInfo::GCCRegAlias> AArch64TargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}

const Builtin::Info AArch64TargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#include "clang/Basic/BuiltinsNEON.def"

#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#include "clang/Basic/BuiltinsAArch64.def"
};
