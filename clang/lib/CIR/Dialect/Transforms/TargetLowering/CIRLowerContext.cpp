//===- CIRLowerContext.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/AST/ASTContext.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRLowerContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include <cmath>

namespace mlir {
namespace cir {

CIRLowerContext::CIRLowerContext(ModuleOp module, clang::LangOptions LOpts)
    : MLIRCtx(module.getContext()), LangOpts(LOpts) {}

CIRLowerContext::~CIRLowerContext() {}

clang::TypeInfo CIRLowerContext::getTypeInfo(Type T) const {
  // TODO(cir): Memoize type info.

  clang::TypeInfo TI = getTypeInfoImpl(T);
  return TI;
}

/// getTypeInfoImpl - Return the size of the specified type, in bits.  This
/// method does not work on incomplete types.
///
/// FIXME: Pointers into different addr spaces could have different sizes and
/// alignment requirements: getPointerInfo should take an AddrSpace, this
/// should take a QualType, &c.
clang::TypeInfo CIRLowerContext::getTypeInfoImpl(const Type T) const {
  uint64_t Width = 0;
  unsigned Align = 8;
  clang::AlignRequirementKind AlignRequirement =
      clang::AlignRequirementKind::None;

  // TODO(cir): We should implement a better way to identify type kinds and use
  // builting data layout interface for this.
  auto typeKind = clang::Type::Builtin;
  if (isa<IntType, SingleType, DoubleType, BoolType>(T)) {
    typeKind = clang::Type::Builtin;
  } else if (isa<StructType>(T)) {
    typeKind = clang::Type::Record;
  } else {
    cir_assert_or_abort(!::cir::MissingFeatures::ABIClangTypeKind(),
                        "Unhandled type class");
    // FIXME(cir): Completely wrong. Just here to make it non-blocking.
    typeKind = clang::Type::Builtin;
  }

  // FIXME(cir): Here we fetch the width and alignment of a type considering the
  // current target. We can likely improve this using MLIR's data layout, or
  // some other interface, to abstract this away (e.g. type.getWidth() &
  // type.getAlign()). Verify if data layout suffices because this would involve
  // some other types such as vectors and complex numbers.
  // FIXME(cir): In the original codegen, this receives an AST type, meaning it
  // differs chars from integers, something that is not possible with the
  // current level of CIR.
  switch (typeKind) {
  case clang::Type::Builtin: {
    if (auto intTy = dyn_cast<IntType>(T)) {
      // NOTE(cir): This assumes int types are already ABI-specific.
      // FIXME(cir): Use data layout interface here instead.
      Width = intTy.getWidth();
      // FIXME(cir): Use the proper getABIAlignment method here.
      Align = std::ceil((float)Width / 8) * 8;
      break;
    }
    if (auto boolTy = dyn_cast<BoolType>(T)) {
      Width = Target->getFloatWidth();
      Align = Target->getFloatAlign();
      break;
    }
    if (auto floatTy = dyn_cast<SingleType>(T)) {
      Width = Target->getFloatWidth();
      Align = Target->getFloatAlign();
      break;
    }
    if (auto doubleTy = dyn_cast<DoubleType>(T)) {
      Width = Target->getDoubleWidth();
      Align = Target->getDoubleAlign();
      break;
    }
    llvm_unreachable("Unknown builtin type!");
    break;
  }
  case clang::Type::Record: {
    const auto RT = dyn_cast<StructType>(T);
    cir_tl_assert(!::cir::MissingFeatures::tagTypeClassAbstraction());

    // Only handle TagTypes (names types) for now.
    cir_tl_assert(RT.getName() && "Anonymous record is NYI");

    // NOTE(cir): Clang does some hanlding of invalid tagged declarations here.
    // Not sure if this is necessary in CIR.

    if (::cir::MissingFeatures::typeGetAsEnumType()) {
      llvm_unreachable("NYI");
    }

    const CIRRecordLayout &Layout = getCIRRecordLayout(RT);
    Width = toBits(Layout.getSize());
    Align = toBits(Layout.getAlignment());
    cir_tl_assert(!::cir::MissingFeatures::recordDeclHasAlignmentAttr());
    break;
  }
  default:
    llvm_unreachable("Unhandled type class");
  }

  cir_tl_assert(llvm::isPowerOf2_32(Align) && "Alignment must be power of 2");
  return clang::TypeInfo(Width, Align, AlignRequirement);
}

Type CIRLowerContext::initBuiltinType(clang::BuiltinType::Kind K) {
  Type Ty;

  // NOTE(cir): Clang does more stuff here. Not sure if we need to do the same.
  cir_tl_assert(!::cir::MissingFeatures::qualifiedTypes());
  switch (K) {
  case clang::BuiltinType::Char_S:
    Ty = IntType::get(getMLIRContext(), 8, true);
    break;
  default:
    llvm_unreachable("NYI");
  }

  Types.push_back(Ty);
  return Ty;
}

void CIRLowerContext::initBuiltinTypes(const clang::TargetInfo &Target,
                                       const clang::TargetInfo *AuxTarget) {
  cir_tl_assert((!this->Target || this->Target == &Target) &&
         "Incorrect target reinitialization");
  this->Target = &Target;
  this->AuxTarget = AuxTarget;

  // C99 6.2.5p3.
  if (LangOpts.CharIsSigned)
    CharTy = initBuiltinType(clang::BuiltinType::Char_S);
  else
    llvm_unreachable("NYI");
}

/// Convert a size in bits to a size in characters.
clang::CharUnits CIRLowerContext::toCharUnitsFromBits(int64_t BitSize) const {
  return clang::CharUnits::fromQuantity(BitSize / getCharWidth());
}

/// Convert a size in characters to a size in characters.
int64_t CIRLowerContext::toBits(clang::CharUnits CharSize) const {
  return CharSize.getQuantity() * getCharWidth();
}

clang::TypeInfoChars CIRLowerContext::getTypeInfoInChars(Type T) const {
  if (auto arrTy = dyn_cast<ArrayType>(T))
    llvm_unreachable("NYI");
  clang::TypeInfo Info = getTypeInfo(T);
  return clang::TypeInfoChars(toCharUnitsFromBits(Info.Width),
                              toCharUnitsFromBits(Info.Align),
                              Info.AlignRequirement);
}

bool CIRLowerContext::isPromotableIntegerType(Type T) const {
  // HLSL doesn't promote all small integer types to int, it
  // just uses the rank-based promotion rules for all types.
  if (::cir::MissingFeatures::langOpts())
    llvm_unreachable("NYI");

  // FIXME(cir): CIR does not distinguish between char, short, etc. So we just
  // assume it is promotable if smaller than 32 bits. This is wrong since, for
  // example, Char32 is promotable. Improve CIR or add an AST query here.
  if (auto intTy = dyn_cast<IntType>(T)) {
    return cast<IntType>(T).getWidth() < 32;
  }

  // Bool are also handled here for codegen parity.
  if (auto boolTy = dyn_cast<BoolType>(T)) {
    return true;
  }

  // Enumerated types are promotable to their compatible integer types
  // (C99 6.3.1.1) a.k.a. its underlying type (C++ [conv.prom]p2).
  // TODO(cir): CIR doesn't know if a integer originated from an enum. Improve
  // CIR or add an AST query here.
  if (::cir::MissingFeatures::typeGetAsEnumType()) {
    llvm_unreachable("NYI");
  }

  return false;
}

} // namespace cir
} // namespace mlir
