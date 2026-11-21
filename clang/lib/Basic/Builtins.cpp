//===--- Builtins.cpp - Builtin function implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements various things for builtin functions.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Builtins.h"
#include "BuiltinTargetFeatures.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/xxhash.h"
#include <memory>
using namespace clang;

const char *HeaderDesc::getName() const {
  switch (ID) {
#define HEADER(ID, NAME)                                                       \
  case ID:                                                                     \
    return NAME;
#include "clang/Basic/BuiltinHeaders.def"
#undef HEADER
  };
  llvm_unreachable("Unknown HeaderDesc::HeaderID enum");
}

static constexpr unsigned NumBuiltins = Builtin::FirstTSBuiltin;

#define GET_BUILTIN_STR_TABLE
#include "clang/Basic/Builtins.inc"
#undef GET_BUILTIN_STR_TABLE

static constexpr Builtin::Info BuiltinInfos[] = {
    Builtin::Info{}, // No-builtin info entry.
#define GET_BUILTIN_INFOS
#include "clang/Basic/Builtins.inc"
#undef GET_BUILTIN_INFOS
};
static_assert(std::size(BuiltinInfos) == NumBuiltins);

/// The shard of the target-independent builtins, which every context has.
static const Builtin::InfosShard TargetIndependentShards[] = {
    {&BuiltinStrings, BuiltinInfos}};

std::pair<const Builtin::InfosShard &, const Builtin::Info &>
Builtin::Context::getShardAndInfo(unsigned ID) const {
  if (Builtins)
    return Builtins->getShardAndInfo(ID);
  assert(ID < Builtin::FirstTSBuiltin && "Invalid builtin ID!");
  return {TargetIndependentShards[0], BuiltinInfos[ID]};
}

/// Return a non-owning StringRef of the builtin's name, reconstructed into Buf.
static StringRef getBuiltinNameInto(const Builtin::InfosShard &Shard,
                                    const Builtin::Info &BuiltinInfo,
                                    SmallVectorImpl<char> &Buf) {
  StringRef Name = (*Shard.Strings)[BuiltinInfo.Offsets.Name];
  if (Shard.NamePrefix.empty())
    return Name;
  Buf.assign(Shard.NamePrefix.begin(), Shard.NamePrefix.end());
  Buf.append(Name.begin(), Name.end());
  return StringRef(Buf.data(), Buf.size());
}

std::string Builtin::Info::getName(const Builtin::InfosShard &Shard) const {
  SmallString<256> Buf;
  return getBuiltinNameInto(Shard, *this, Buf).str();
}

/// Return the identifier name for the specified builtin,
/// e.g. "__builtin_abs".
std::string Builtin::Context::getName(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return I.getName(Shard);
}

SmallString<64> Builtin::Context::getQuotedName(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  SmallString<64> Result("'");
  Result += Shard.NamePrefix;
  Result += (*Shard.Strings)[I.Offsets.Name];
  Result += '\'';
  return Result;
}

const char *Builtin::Context::getTypeString(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return (*Shard.Strings)[I.Offsets.Type].data();
}

const char *Builtin::Context::getAttributesString(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return (*Shard.Strings)[I.Offsets.Attributes].data();
}

const char *Builtin::Context::getRequiredFeatures(unsigned ID) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return (*Shard.Strings)[I.Offsets.Features].data();
}

Builtin::Context::Context() = default;

Builtin::Context::Context(const TargetInfo &Target, const TargetInfo *AuxTarget)
    : Context(std::make_shared<const TargetBuiltins>(Target, AuxTarget)) {}

Builtin::Context::Context(std::shared_ptr<const TargetBuiltins> Builtins)
    : Builtins(std::move(Builtins)),
      NumTargetBuiltins(this->Builtins->getNumTargetBuiltins()) {}

bool Builtin::Context::isBuiltinFunc(llvm::StringRef FuncName) {
  bool InStdNamespace = FuncName.consume_front("std-");
  for (const auto &Shard : TargetIndependentShards)
    if (llvm::StringRef FuncNameSuffix = FuncName;
        FuncNameSuffix.consume_front(Shard.NamePrefix))
      for (const auto &I : Shard.Infos)
        if (FuncNameSuffix == (*Shard.Strings)[I.Offsets.Name] &&
            (bool)strchr((*Shard.Strings)[I.Offsets.Attributes].data(), 'z') ==
                InStdNamespace)
          return strchr((*Shard.Strings)[I.Offsets.Attributes].data(), 'f') !=
                 nullptr;

  return false;
}

/// Is this builtin supported according to the given language options?
static bool builtinIsSupported(const llvm::StringTable &Strings,
                               const Builtin::Info &BuiltinInfo,
                               const LangOptions &LangOpts) {
  auto AttributesStr = Strings[BuiltinInfo.Offsets.Attributes];

  /* Builtins Unsupported */
  if (LangOpts.NoBuiltin && strchr(AttributesStr.data(), 'f') != nullptr)
    return false;
  /* CorBuiltins Unsupported */
  if (!LangOpts.Coroutines && (BuiltinInfo.Langs & COR_LANG))
    return false;
  /* MathBuiltins Unsupported */
  if (LangOpts.NoMathBuiltin && BuiltinInfo.Header.ID == HeaderDesc::MATH_H)
    return false;
  /* GnuMode Unsupported */
  if (!LangOpts.GNUMode && (BuiltinInfo.Langs & GNU_LANG))
    return false;
  /* MSMode Unsupported */
  if (!LangOpts.MicrosoftExt && (BuiltinInfo.Langs & MS_LANG))
    return false;
  /* HLSLMode Unsupported */
  if (!LangOpts.HLSL && (BuiltinInfo.Langs & HLSL_LANG))
    return false;
  /* ObjC Unsupported */
  if (!LangOpts.ObjC && BuiltinInfo.Langs == OBJC_LANG)
    return false;
  /* OpenCLC Unsupported */
  if (!LangOpts.OpenCL && (BuiltinInfo.Langs & ALL_OCL_LANGUAGES))
    return false;
  /* OpenCL GAS Unsupported */
  if (!LangOpts.OpenCLGenericAddressSpace && (BuiltinInfo.Langs & OCL_GAS))
    return false;
  /* OpenCL Pipe Unsupported */
  if (!LangOpts.OpenCLPipes && (BuiltinInfo.Langs & OCL_PIPE))
    return false;

  // Device side enqueue is not supported until OpenCL 2.0. In 2.0 and higher
  // support is indicated with language option for blocks.

  /* OpenCL DSE Unsupported */
  if ((LangOpts.getOpenCLCompatibleVersion() < 200 || !LangOpts.Blocks) &&
      (BuiltinInfo.Langs & OCL_DSE))
    return false;
  /* OpenMP Unsupported */
  if (!LangOpts.OpenMP && BuiltinInfo.Langs == OMP_LANG)
    return false;
  /* CUDA Unsupported */
  if (!LangOpts.CUDA && BuiltinInfo.Langs == CUDA_LANG)
    return false;
  /* CPlusPlus Unsupported */
  if (!LangOpts.CPlusPlus && BuiltinInfo.Langs == CXX_LANG)
    return false;
  /* consteval Unsupported */
  if (!LangOpts.CPlusPlus20 && strchr(AttributesStr.data(), 'G') != nullptr)
    return false;
  /* C23 unsupported */
  if (!LangOpts.C23 && BuiltinInfo.Langs == C23_LANG)
    return false;
  /* C2y unsupported */
  if (!LangOpts.C2y && BuiltinInfo.Langs == C2Y_LANG)
    return false;
  /* Reflection unsupported */
  if (!LangOpts.Reflection && (BuiltinInfo.Langs & REFLECTION_LANG))
    return false;
  return true;
}

static bool isBuiltinConstForTriple(unsigned BuiltinID, llvm::Triple Trip) {
  // There's a special case with the fma builtins where they are always const
  // if the target environment is GNU or the target is OS is Windows and we're
  // targeting the MSVCRT.dll environment.
  // FIXME: This list can be become outdated. Need to find a way to get it some
  // other way.
  switch (BuiltinID) {
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmal:
  case Builtin::BI__builtin_fmaf16:
  case Builtin::BIfma:
  case Builtin::BIfmaf:
  case Builtin::BIfmal: {
    if (Trip.isGNUEnvironment() || Trip.isOSMSVCRT())
      return true;
    break;
  }
  default:
    break;
  }

  return false;
}

bool Builtin::Context::shouldGenerateFPMathIntrinsic(
    unsigned BuiltinID, llvm::Triple Trip, std::optional<bool> ErrnoOverwritten,
    bool MathErrnoEnabled, bool HasOptNoneAttr,
    bool IsOptimizationEnabled) const {

  // True if we are compiling at -O2 and errno has been disabled
  // using the '#pragma float_control(precise, off)', and
  // attribute opt-none hasn't been seen.
  bool ErrnoOverridenToFalseWithOpt = ErrnoOverwritten.has_value() &&
                                      !ErrnoOverwritten.value() &&
                                      !HasOptNoneAttr && IsOptimizationEnabled;

  // There are LLVM math intrinsics/instructions corresponding to math library
  // functions except the LLVM op will never set errno while the math library
  // might. Also, math builtins have the same semantics as their math library
  // twins. Thus, we can transform math library and builtin calls to their
  // LLVM counterparts if the call is marked 'const' (known to never set errno).
  // In case FP exceptions are enabled, the experimental versions of the
  // intrinsics model those.
  bool ConstAlways =
      isConst(BuiltinID) || isBuiltinConstForTriple(BuiltinID, Trip);

  bool ConstWithoutErrnoAndExceptions =
      isConstWithoutErrnoAndExceptions(BuiltinID);
  bool ConstWithoutExceptions = isConstWithoutExceptions(BuiltinID);

  // ConstAttr is enabled in fast-math mode. In fast-math mode, math-errno is
  // disabled.
  // Math intrinsics are generated only when math-errno is disabled. Any pragmas
  // or attributes that affect math-errno should prevent or allow math
  // intrinsics to be generated. Intrinsics are generated:
  //   1- In fast math mode, unless math-errno is overriden
  //      via '#pragma float_control(precise, on)', or via an
  //      'attribute__((optnone))'.
  //   2- If math-errno was enabled on command line but overriden
  //      to false via '#pragma float_control(precise, off))' and
  //      'attribute__((optnone))' hasn't been used.
  //   3- If we are compiling with optimization and errno has been disabled
  //      via '#pragma float_control(precise, off)', and
  //      'attribute__((optnone))' hasn't been used.

  bool ConstWithoutErrnoOrExceptions =
      ConstWithoutErrnoAndExceptions || ConstWithoutExceptions;
  bool GenerateIntrinsics =
      (ConstAlways && !HasOptNoneAttr) ||
      (!MathErrnoEnabled &&
       !(ErrnoOverwritten.has_value() && ErrnoOverwritten.value()) &&
       !HasOptNoneAttr);
  if (!GenerateIntrinsics) {
    GenerateIntrinsics =
        ConstWithoutErrnoOrExceptions && !ConstWithoutErrnoAndExceptions;
    if (!GenerateIntrinsics)
      GenerateIntrinsics =
          ConstWithoutErrnoOrExceptions &&
          (!MathErrnoEnabled &&
           !(ErrnoOverwritten.has_value() && ErrnoOverwritten.value()) &&
           !HasOptNoneAttr);
    if (!GenerateIntrinsics)
      GenerateIntrinsics =
          ConstWithoutErrnoOrExceptions && ErrnoOverridenToFalseWithOpt;
  }

  return GenerateIntrinsics;
}

static uint32_t hashBuiltinName(StringRef Name) {
  return static_cast<uint32_t>(llvm::xxh3_64bits(Name));
}

Builtin::TargetBuiltins::TargetBuiltins(const TargetInfo &Target,
                                        const TargetInfo *AuxTarget) {
  llvm::append_range(Shards, TargetIndependentShards);
  llvm::append_range(Shards, Target.getTargetBuiltins());
  NumTargetShards = Shards.size() - 1;
  if (AuxTarget)
    llvm::append_range(Shards, AuxTarget->getTargetBuiltins());
  for (const InfosShard &Shard : getTargetShards())
    NumTargetBuiltins += Shard.Infos.size();
  for (const InfosShard &Shard : getAuxTargetShards())
    NumAuxTargetBuiltins += Shard.Infos.size();

  size_t NumNames = 0;
  for (const InfosShard &Shard : Shards)
    NumNames += Shard.Infos.size();
  // Keep the load factor at or below 1/2.
  size_t NumSlots = llvm::PowerOf2Ceil(NumNames * 2);
  Slots = std::make_unique<uint64_t[]>(NumSlots);
  Mask = NumSlots - 1;
  unsigned ID = 0;
  SmallString<256> NameBuf;
  for (const InfosShard &Shard : Shards) {
    for (const Info &I : Shard.Infos) {
      // Skip the "not a builtin" entry.
      if (ID != NotBuiltin) {
        uint32_t Hash = hashBuiltinName(getBuiltinNameInto(Shard, I, NameBuf));
        size_t Slot = Hash & Mask;
        while (Slots[Slot])
          Slot = (Slot + 1) & Mask;
        Slots[Slot] = (uint64_t(Hash) << 32) | ID;
      }
      ++ID;
    }
  }
}

std::pair<const Builtin::InfosShard &, const Builtin::Info &>
Builtin::TargetBuiltins::getShardAndInfo(unsigned ID) const {
  assert(ID < FirstTSBuiltin + NumTargetBuiltins + NumAuxTargetBuiltins &&
         "Invalid builtin ID!");
  // Find the group of shards from the ID, then loop over the shards of the
  // group to find the one matching this ID. We don't expect to have many
  // shards and so its better to search linearly than with a binary search.
  ArrayRef<InfosShard> Group = ArrayRef(Shards).take_front(1);
  if (ID >= FirstTSBuiltin + NumTargetBuiltins) {
    Group = getAuxTargetShards();
    ID -= FirstTSBuiltin + NumTargetBuiltins;
  } else if (ID >= FirstTSBuiltin) {
    Group = getTargetShards();
    ID -= FirstTSBuiltin;
  }
  for (const InfosShard &Shard : Group) {
    if (ID < Shard.Infos.size())
      return {Shard, Shard.Infos[ID]};
    ID -= Shard.Infos.size();
  }
  llvm_unreachable("Invalid target builtin shard structure!");
}

bool Builtin::TargetBuiltins::hasName(unsigned ID, StringRef Name) const {
  const auto &[Shard, I] = getShardAndInfo(ID);
  return Name.consume_front(Shard.NamePrefix) &&
         Name == (*Shard.Strings)[I.Offsets.Name];
}

SmallVector<unsigned, 2> Builtin::TargetBuiltins::lookup(StringRef Name) const {
  SmallVector<unsigned, 2> IDs;
  uint32_t Hash = hashBuiltinName(Name);
  for (size_t Slot = Hash & Mask; uint64_t Entry = Slots[Slot];
       Slot = (Slot + 1) & Mask) {
    unsigned ID = static_cast<uint32_t>(Entry);
    if (Entry >> 32 == Hash && hasName(ID, Name))
      IDs.push_back(ID);
  }
  return IDs;
}

unsigned Builtin::Context::lookupBuiltinID(StringRef Name,
                                           const LangOptions &LangOpts) const {
  // This needs to behave as if the builtins were registered in the order of
  // their IDs, with later registrations overriding earlier ones: aux target
  // builtins win over target builtins, which win over target-independent
  // ones. Aux target builtins are registered even if unsupported.
  if (!Builtins)
    return NotBuiltin;
  unsigned Result = NotBuiltin;
  for (unsigned ID : Builtins->lookup(Name)) {
    if (ID <= Result)
      continue;
    const auto &[Shard, I] = getShardAndInfo(ID);
    if (isAuxBuiltinID(ID) || builtinIsSupported(*Shard.Strings, I, LangOpts))
      Result = ID;
  }
  if (Result == NotBuiltin)
    return NotBuiltin;

  // Library builtins can be disabled with -fno-builtin-foo, or with
  // -fno-builtin-std-foo for those in namespace std.
  if (isPredefinedLibFunction(Result)) {
    bool InStdNamespace = isInStdNamespace(Result);
    for (StringRef NoBuiltin : LangOpts.NoBuiltinFuncs)
      if (NoBuiltin.consume_front("std-") == InStdNamespace &&
          NoBuiltin == Name)
        return NotBuiltin;
  }
  return Result;
}

void Builtin::Context::forEachBuiltinName(
    const LangOptions &LangOpts, llvm::function_ref<void(StringRef)> Fn) const {
  if (!Builtins)
    return;
  llvm::SmallString<256> NameBuf;
  // Skip the "not a builtin" entry.
  for (const auto &Shard : TargetIndependentShards)
    for (const auto &I : Shard.Infos.drop_front())
      if (builtinIsSupported(*Shard.Strings, I, LangOpts))
        Fn(getBuiltinNameInto(Shard, I, NameBuf));
  for (const auto &Shard : Builtins->getTargetShards())
    for (const auto &I : Shard.Infos)
      if (builtinIsSupported(*Shard.Strings, I, LangOpts))
        Fn(getBuiltinNameInto(Shard, I, NameBuf));
  for (const auto &Shard : Builtins->getAuxTargetShards())
    for (const auto &I : Shard.Infos)
      Fn(getBuiltinNameInto(Shard, I, NameBuf));
}

unsigned Builtin::Context::getRequiredVectorWidth(unsigned ID) const {
  const char *WidthPos = ::strchr(getAttributesString(ID), 'V');
  if (!WidthPos)
    return 0;

  ++WidthPos;
  assert(*WidthPos == ':' &&
         "Vector width specifier must be followed by a ':'");
  ++WidthPos;

  char *EndPos;
  unsigned Width = ::strtol(WidthPos, &EndPos, 10);
  assert(*EndPos == ':' && "Vector width specific must end with a ':'");
  return Width;
}

bool Builtin::Context::isLike(unsigned ID, unsigned &FormatIdx,
                              bool &HasVAListArg, const char *Fmt) const {
  assert(Fmt && "Not passed a format string");
  assert(::strlen(Fmt) == 2 &&
         "Format string needs to be two characters long");
  assert(::toupper(Fmt[0]) == Fmt[1] &&
         "Format string is not in the form \"xX\"");

  const char *Like = ::strpbrk(getAttributesString(ID), Fmt);
  if (!Like)
    return false;

  HasVAListArg = (*Like == Fmt[1]);

  ++Like;
  assert(*Like == ':' && "Format specifier must be followed by a ':'");
  ++Like;

  assert(::strchr(Like, ':') && "Format specifier must end with a ':'");
  FormatIdx = ::strtol(Like, nullptr, 10);
  return true;
}

bool Builtin::Context::isPrintfLike(unsigned ID, unsigned &FormatIdx,
                                    bool &HasVAListArg) {
  return isLike(ID, FormatIdx, HasVAListArg, "pP");
}

bool Builtin::Context::isScanfLike(unsigned ID, unsigned &FormatIdx,
                                   bool &HasVAListArg) {
  return isLike(ID, FormatIdx, HasVAListArg, "sS");
}

static void parseCommaSeparatedIndices(const char *CurrPos,
                                       llvm::SmallVectorImpl<int> &Indxs) {
  assert(*CurrPos == '<' && "Expected '<' to start index list");
  ++CurrPos;

  char *EndPos;
  int PosIdx = ::strtol(CurrPos, &EndPos, 10);
  assert(PosIdx >= 0 && "Index is supposed to be positive!");
  Indxs.push_back(PosIdx);

  while (*EndPos == ',') {
    const char *PayloadPos = EndPos + 1;

    int PayloadIdx = ::strtol(PayloadPos, &EndPos, 10);
    Indxs.push_back(PayloadIdx);
  }

  assert(*EndPos == '>' && "Index list must end with '>'");
}

bool Builtin::Context::isNonNull(unsigned ID, llvm::SmallVectorImpl<int> &Indxs,
                                 Info::NonNullMode &Mode) const {

  const char *AttrPos = ::strchr(getAttributesString(ID), 'N');
  if (!AttrPos)
    return false;

  ++AttrPos;
  assert(*AttrPos == ':' && "Format specifier must be followed by a ':'");
  ++AttrPos;
  if (*AttrPos == '0')
    Mode = Info::NonNullMode::NonOptimizing;
  else if (*AttrPos == '1')
    Mode = Info::NonNullMode::Optimizing;
  else
    llvm_unreachable("Unrecognized NonNull optimization mode");
  ++AttrPos; // skip mode
  assert(*AttrPos == ':' && "Mode must be followed by a ':'");
  ++AttrPos;

  parseCommaSeparatedIndices(AttrPos, Indxs);

  return true;
}

void Builtin::Context::getArgRanges(
    unsigned ID, llvm::SmallVectorImpl<ArgRange> &Ranges) const {
  StringRef Attrs = getAttributesString(ID);
  size_t Pos = Attrs.find('R');
  if (Pos == StringRef::npos)
    return;

  StringRef Str = Attrs.drop_front(Pos + 1);
  if (!Str.consume_front("<"))
    llvm_unreachable("Expected '<' to start range list");

  do {
    ArgRange Range;
    if (Str.consumeInteger(10, Range.Index) || !Str.consume_front(":") ||
        Str.consumeInteger(10, Range.Lower) || !Str.consume_front(":") ||
        Str.consumeInteger(10, Range.Upper))
      llvm_unreachable("Malformed argument range");
    Ranges.push_back(Range);
  } while (Str.consume_front(","));

  assert(Str.starts_with('>') && "Range list must end with '>'");
}

bool Builtin::Context::performsCallback(unsigned ID,
                                        SmallVectorImpl<int> &Encoding) const {
  const char *CalleePos = ::strchr(getAttributesString(ID), 'C');
  if (!CalleePos)
    return false;

  ++CalleePos;
  parseCommaSeparatedIndices(CalleePos, Encoding);

  return true;
}

bool Builtin::Context::canBeRedeclared(unsigned ID) const {
  return ID == Builtin::NotBuiltin || ID == Builtin::BI__va_start ||
         ID == Builtin::BI__builtin_assume_aligned ||
         (!hasReferenceArgsOrResult(ID) && !hasCustomTypechecking(ID)) ||
         isInStdNamespace(ID);
}

bool Builtin::evaluateRequiredTargetFeatures(
    StringRef RequiredFeatures, const llvm::StringMap<bool> &TargetFetureMap) {
  // Return true if the builtin doesn't have any required features.
  if (RequiredFeatures.empty())
    return true;
  assert(!RequiredFeatures.contains(' ') && "Space in feature list");

  TargetFeatures TF(TargetFetureMap);
  return TF.hasRequiredFeatures(RequiredFeatures);
}
