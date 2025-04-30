//===--- Next32.cpp - Implement Next32 target feature support
//---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Next32 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

Next32TargetInfo::Next32TargetInfo(const llvm::Triple &Triple,
                                   const TargetOptions &)
    : TargetInfo(Triple) {
  TLSSupported = true;
  VLASupported = true;
  NoAsmVariants = true;
  LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
  RegParmMax =
      6; // Can't be larger or clang breaks in TargetInfo::getRegParmMax()

  assert(!BigEndian && "Big endian Next32 triples not supported.");

  resetDataLayout("e-S128-m:e-n8:16:32-i64:64");

  MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
  static_assert(std::size(Next32AddrSpaceMap) ==
                    static_cast<size_t>(LangAS::FirstTargetAddressSpace),
                "Number of entires in Next32AddrSpaceMap doesn't coresponds "
                "number of address spaces.");
  AddrSpaceMap = &Next32AddrSpaceMap;
}

static constexpr llvm::StringLiteral ValidCPUNames[] = {{"next32gen1"},
                                                        {"next32gen2"}};

bool Next32TargetInfo::isValidCPUName(StringRef Name) const {
  return llvm::find(ValidCPUNames, Name) != std::end(ValidCPUNames);
}

void Next32TargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  Values.append(std::begin(ValidCPUNames), std::end(ValidCPUNames));
}

void Next32TargetInfo::getTargetDefines(const LangOptions &Opts,
                                        MacroBuilder &Builder) const {
  Builder.defineMacro("__NEXT32__");
  Builder.defineMacro("__NEXT__");
  // Define __NO_MATH_INLINES on Next32 so that we don't get inline
  // functions in glibc header files that use FP Stack inline asm which the
  // backend can't deal with.
  Builder.defineMacro("__NO_MATH_INLINES");

  // Hack: define x86_64 to ensure matching the host architecture sizes
  Builder.defineMacro("__amd64__");
  Builder.defineMacro("__amd64");
  Builder.defineMacro("__x86_64__");
  Builder.defineMacro("__x86_64");
}

const Builtin::Info Next32TargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
  {#ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr},
#include "clang/Basic/BuiltinsNext32.def"
};

ArrayRef<Builtin::Info> Next32TargetInfo::getTargetBuiltins() const {
  return llvm::ArrayRef(BuiltinInfo,
                        clang::Next32::LastTSBuiltin - Builtin::FirstTSBuiltin);
}

bool Next32TargetInfo::isValidFeatureName(StringRef Name) const {
  return llvm::StringSwitch<bool>(Name).Default(false);
}

bool Next32TargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature).Default(false);
}

bool Next32TargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                            DiagnosticsEngine &Diags) {
  for (const auto &Feature : Features) {
    if ((Feature[0] != '+') && (Feature[0] != '-'))
      continue;
  }

  return true;
}
