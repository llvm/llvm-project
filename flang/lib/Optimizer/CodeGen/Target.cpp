//===-- Target.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "Target.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/Triple.h"

#define DEBUG_TYPE "flang-codegen-target"

using namespace fir;

// Reduce a REAL/float type to the floating point semantics.
static const llvm::fltSemantics &floatToSemantics(KindMapping &kindMap,
                                                  mlir::Type type) {
  assert(isa_real(type));
  if (auto ty = type.dyn_cast<fir::RealType>())
    return kindMap.getFloatSemantics(ty.getFKind());
  return type.cast<mlir::FloatType>().getFloatSemantics();
}

namespace {
template <typename S>
struct GenericTarget : public CodeGenSpecifics {
  using CodeGenSpecifics::CodeGenSpecifics;
  using AT = CodeGenSpecifics::Attributes;

  mlir::Type complexMemoryType(mlir::Type eleTy) const override {
    assert(fir::isa_real(eleTy));
    // { t, t }   struct of 2 eleTy
    mlir::TypeRange range = {eleTy, eleTy};
    return mlir::TupleType::get(range, eleTy.getContext());
  }

  mlir::Type boxcharMemoryType(mlir::Type eleTy) const override {
    auto idxTy = mlir::IntegerType::get(S::defaultWidth, eleTy.getContext());
    auto ptrTy = fir::ReferenceType::get(eleTy);
    // { t*, index }
    mlir::TypeRange range = {ptrTy, idxTy};
    return mlir::TupleType::get(range, eleTy.getContext());
  }

  Marshalling boxcharArgumentType(mlir::Type eleTy, bool sret) const override {
    CodeGenSpecifics::Marshalling marshal;
    auto idxTy = mlir::IntegerType::get(S::defaultWidth, eleTy.getContext());
    auto ptrTy = fir::ReferenceType::get(eleTy);
    marshal.emplace_back(ptrTy, AT{});
    // Return value arguments are grouped as a pair. Others are passed in a
    // split format with all pointers first (in the declared position) and all
    // LEN arguments appended after all of the dummy arguments.
    // NB: Other conventions/ABIs can/should be supported via options.
    marshal.emplace_back(idxTy, AT{0, {}, {}, /*append=*/!sret});
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// i386 (x86 32 bit) linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetI386 : public GenericTarget<TargetI386> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 32;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Type eleTy) const override {
    assert(fir::isa_real(eleTy));
    CodeGenSpecifics::Marshalling marshal;
    // { t, t }   struct of 2 eleTy, byval, align 4
    mlir::TypeRange range = {eleTy, eleTy};
    auto structTy = mlir::TupleType::get(range, eleTy.getContext());
    marshal.emplace_back(fir::ReferenceType::get(structTy),
                         AT{4, /*byval=*/true, {}});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Type eleTy) const override {
    assert(fir::isa_real(eleTy));
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle()) {
      // i64   pack both floats in a 64-bit GPR
      marshal.emplace_back(mlir::IntegerType::get(64, eleTy.getContext()),
                           AT{});
    } else if (sem == &llvm::APFloat::IEEEdouble()) {
      // { t, t }   struct of 2 eleTy, sret, align 4
      mlir::TypeRange range = {eleTy, eleTy};
      auto structTy = mlir::TupleType::get(range, eleTy.getContext());
      marshal.emplace_back(fir::ReferenceType::get(structTy),
                           AT{4, {}, /*sret=*/true});
    } else {
      llvm_unreachable("not implemented");
    }
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// x86_64 (x86 64 bit) linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetX86_64 : public GenericTarget<TargetX86_64> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle()) {
      // <2 x t>   vector of 2 eleTy
      marshal.emplace_back(fir::VectorType::get(2, eleTy), AT{});
    } else if (sem == &llvm::APFloat::IEEEdouble()) {
      // two distinct double arguments
      marshal.emplace_back(eleTy, AT{});
      marshal.emplace_back(eleTy, AT{});
    } else {
      llvm_unreachable("not implemented");
    }
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle()) {
      // <2 x t>   vector of 2 eleTy
      marshal.emplace_back(fir::VectorType::get(2, eleTy), AT{});
    } else if (sem == &llvm::APFloat::IEEEdouble()) {
      // { double, double }   struct of 2 double
      mlir::TypeRange range = {eleTy, eleTy};
      marshal.emplace_back(mlir::TupleType::get(range, eleTy.getContext()),
                           AT{});
    } else {
      llvm_unreachable("not implemented");
    }
    return marshal;
  }
};
} // namespace

// Instantiate the overloaded target instance based on the triple value.
// Currently, the implementation only instantiates `i386-unknown-linux-gnu` and
// `x86_64-unknown-linux-gnu` like triples. Other targets should be added to
// this file as needed.
std::unique_ptr<fir::CodeGenSpecifics>
fir::CodeGenSpecifics::get(mlir::MLIRContext *ctx, llvm::Triple &trp,
                           KindMapping &kindMap) {
  switch (trp.getArch()) {
  default:
    break;
  case llvm::Triple::ArchType::x86:
    switch (trp.getOS()) {
    default:
      break;
    case llvm::Triple::OSType::Linux:
      return std::make_unique<TargetI386>(ctx, trp, kindMap);
    }
    break;
  case llvm::Triple::ArchType::x86_64:
    switch (trp.getOS()) {
    default:
      break;
    case llvm::Triple::OSType::Linux:
      return std::make_unique<TargetX86_64>(ctx, trp, kindMap);
    }
    break;
  }
  llvm::report_fatal_error("target not implemented");
}
