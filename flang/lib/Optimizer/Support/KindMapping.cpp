//===-- KindMapping.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/CommandLine.h"

/// Allow the user to set the FIR intrinsic type kind value to LLVM type
/// mappings.  Note that these are not mappings from kind values to any
/// other MLIR dialect, only to LLVM IR. The default values follow the f18
/// front-end kind mappings.

using Bitsize = fir::KindMapping::Bitsize;
using KindTy = fir::KindMapping::KindTy;
using LLVMTypeID = fir::KindMapping::LLVMTypeID;
using MatchResult = fir::KindMapping::MatchResult;

static llvm::cl::opt<std::string> clKindMapping(
    "kind-mapping", llvm::cl::desc("kind mapping string to set kind precision"),
    llvm::cl::value_desc("kind-mapping-string"), llvm::cl::init(""));

/// Integral types default to the kind value being the size of the value in
/// bytes. The default is to scale from bytes to bits.
static Bitsize defaultScalingKind(KindTy kind) {
  const unsigned BITS_IN_BYTE = 8;
  return kind * BITS_IN_BYTE;
}

/// Floating-point types default to the kind value being the size of the value
/// in bytes. The default is to translate kinds of 2, 4, 8, 10, and 16 to a
/// valid llvm::Type::TypeID value. Otherwise, the default is FloatTyID.
static LLVMTypeID defaultRealKind(KindTy kind) {
  switch (kind) {
  case 2:
    return LLVMTypeID::HalfTyID;
  case 4:
    return LLVMTypeID::FloatTyID;
  case 8:
    return LLVMTypeID::DoubleTyID;
  case 10:
    return LLVMTypeID::X86_FP80TyID;
  case 16:
    return LLVMTypeID::FP128TyID;
  default:
    return LLVMTypeID::FloatTyID;
  }
}

// lookup the kind-value given the defaults, the mappings, and a KIND key
template <typename RT, char KEY>
static RT doLookup(std::function<RT(KindTy)> def,
                   const llvm::DenseMap<std::pair<char, KindTy>, RT> &map,
                   KindTy kind) {
  std::pair<char, KindTy> key{KEY, kind};
  auto iter = map.find(key);
  if (iter != map.end())
    return iter->second;
  return def(kind);
}

// do a lookup for INTERGER, LOGICAL, or CHARACTER
template <char KEY, typename MAP>
static Bitsize getIntegerLikeBitsize(KindTy kind, const MAP &map) {
  return doLookup<Bitsize, KEY>(defaultScalingKind, map, kind);
}

// do a lookup for REAL or COMPLEX
template <char KEY, typename MAP>
static LLVMTypeID getFloatLikeTypeID(KindTy kind, const MAP &map) {
  return doLookup<LLVMTypeID, KEY>(defaultRealKind, map, kind);
}

template <char KEY, typename MAP>
static const llvm::fltSemantics &getFloatSemanticsOfKind(KindTy kind,
                                                         const MAP &map) {
  switch (doLookup<LLVMTypeID, KEY>(defaultRealKind, map, kind)) {
  case LLVMTypeID::HalfTyID:
    return llvm::APFloat::IEEEhalf();
  case LLVMTypeID::FloatTyID:
    return llvm::APFloat::IEEEsingle();
  case LLVMTypeID::DoubleTyID:
    return llvm::APFloat::IEEEdouble();
  case LLVMTypeID::X86_FP80TyID:
    return llvm::APFloat::x87DoubleExtended();
  case LLVMTypeID::FP128TyID:
    return llvm::APFloat::IEEEquad();
  case LLVMTypeID::PPC_FP128TyID:
    return llvm::APFloat::PPCDoubleDouble();
  default:
    llvm_unreachable("Invalid floating type");
  }
}

static MatchResult parseCode(char &code, const char *&ptr) {
  if (*ptr != 'a' && *ptr != 'c' && *ptr != 'i' && *ptr != 'l' && *ptr != 'r')
    return mlir::failure();
  code = *ptr++;
  return mlir::success();
}

template <char ch>
static MatchResult parseSingleChar(const char *&ptr) {
  if (*ptr != ch)
    return mlir::failure();
  ++ptr;
  return mlir::success();
}

static MatchResult parseColon(const char *&ptr) {
  return parseSingleChar<':'>(ptr);
}

static MatchResult parseComma(const char *&ptr) {
  return parseSingleChar<','>(ptr);
}

static MatchResult parseInt(unsigned &result, const char *&ptr) {
  const char *beg = ptr;
  while (*ptr >= '0' && *ptr <= '9')
    ptr++;
  if (beg == ptr)
    return mlir::failure();
  llvm::StringRef ref(beg, ptr - beg);
  int temp;
  if (ref.consumeInteger(10, temp))
    return mlir::failure();
  result = temp;
  return mlir::success();
}

static mlir::LogicalResult matchString(const char *&ptr,
                                       llvm::StringRef literal) {
  llvm::StringRef s(ptr);
  if (s.startswith(literal)) {
    ptr += literal.size();
    return mlir::success();
  }
  return mlir::failure();
}

static MatchResult parseTypeID(LLVMTypeID &result, const char *&ptr) {
  if (mlir::succeeded(matchString(ptr, "Half"))) {
    result = LLVMTypeID::HalfTyID;
    return mlir::success();
  }
  if (mlir::succeeded(matchString(ptr, "Float"))) {
    result = LLVMTypeID::FloatTyID;
    return mlir::success();
  }
  if (mlir::succeeded(matchString(ptr, "Double"))) {
    result = LLVMTypeID::DoubleTyID;
    return mlir::success();
  }
  if (mlir::succeeded(matchString(ptr, "X86_FP80"))) {
    result = LLVMTypeID::X86_FP80TyID;
    return mlir::success();
  }
  if (mlir::succeeded(matchString(ptr, "FP128"))) {
    result = LLVMTypeID::FP128TyID;
    return mlir::success();
  }
  if (mlir::succeeded(matchString(ptr, "PPC_FP128"))) {
    result = LLVMTypeID::PPC_FP128TyID;
    return mlir::success();
  }
  return mlir::failure();
}

fir::KindMapping::KindMapping(mlir::MLIRContext *context, llvm::StringRef map)
    : context{context} {
  if (mlir::failed(parse(map))) {
    intMap.clear();
    floatMap.clear();
  }
}

fir::KindMapping::KindMapping(mlir::MLIRContext *context)
    : KindMapping{context, clKindMapping} {}

MatchResult fir::KindMapping::badMapString(const llvm::Twine &ptr) {
  auto unknown = mlir::UnknownLoc::get(context);
  mlir::emitError(unknown, ptr);
  return mlir::failure();
}

MatchResult fir::KindMapping::parse(llvm::StringRef kindMap) {
  if (kindMap.empty())
    return mlir::success();
  const char *srcPtr = kindMap.begin();
  while (true) {
    char code = '\0';
    KindTy kind = 0;
    if (parseCode(code, srcPtr) || parseInt(kind, srcPtr))
      return badMapString(srcPtr);
    if (code == 'a' || code == 'i' || code == 'l') {
      Bitsize bits = 0;
      if (parseColon(srcPtr) || parseInt(bits, srcPtr))
        return badMapString(srcPtr);
      intMap[std::pair<char, KindTy>{code, kind}] = bits;
    } else if (code == 'r' || code == 'c') {
      LLVMTypeID id{};
      if (parseColon(srcPtr) || parseTypeID(id, srcPtr))
        return badMapString(srcPtr);
      floatMap[std::pair<char, KindTy>{code, kind}] = id;
    } else {
      return badMapString(srcPtr);
    }
    if (parseComma(srcPtr))
      break;
  }
  if (*srcPtr)
    return badMapString(srcPtr);
  return mlir::success();
}

Bitsize fir::KindMapping::getCharacterBitsize(KindTy kind) const {
  return getIntegerLikeBitsize<'a'>(kind, intMap);
}

Bitsize fir::KindMapping::getIntegerBitsize(KindTy kind) const {
  return getIntegerLikeBitsize<'i'>(kind, intMap);
}

Bitsize fir::KindMapping::getLogicalBitsize(KindTy kind) const {
  return getIntegerLikeBitsize<'l'>(kind, intMap);
}

LLVMTypeID fir::KindMapping::getRealTypeID(KindTy kind) const {
  return getFloatLikeTypeID<'r'>(kind, floatMap);
}

LLVMTypeID fir::KindMapping::getComplexTypeID(KindTy kind) const {
  return getFloatLikeTypeID<'c'>(kind, floatMap);
}

Bitsize fir::KindMapping::getRealBitsize(KindTy kind) const {
  auto typeId = getFloatLikeTypeID<'r'>(kind, floatMap);
  llvm::LLVMContext llCtxt; // FIXME
  return llvm::Type::getPrimitiveType(llCtxt, typeId)->getPrimitiveSizeInBits();
}

const llvm::fltSemantics &
fir::KindMapping::getFloatSemantics(KindTy kind) const {
  return getFloatSemanticsOfKind<'r'>(kind, floatMap);
}
