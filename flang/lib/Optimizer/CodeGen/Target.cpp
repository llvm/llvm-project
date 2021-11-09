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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"

#define DEBUG_TYPE "flang-codegen-target"

using namespace fir;

namespace {
template <typename S>
struct GenericTarget : public CodeGenSpecifics {
  using CodeGenSpecifics::CodeGenSpecifics;
  using AT = CodeGenSpecifics::Attributes;

  Marshalling boxcharArgumentType(mlir::Type eleTy, bool sret) const override {
    CodeGenSpecifics::Marshalling marshal;
    auto idxTy = mlir::IntegerType::get(eleTy.getContext(), S::defaultWidth);
    auto ptrTy = fir::ReferenceType::get(eleTy);
    marshal.emplace_back(ptrTy, AT{});
    // Return value arguments are grouped as a pair. Others are passed in a
    // split format with all pointers first (in the declared position) and all
    // LEN arguments appended after all of the dummy arguments.
    // NB: Other conventions/ABIs can/should be supported via options.
    marshal.emplace_back(idxTy, AT{/*append=*/!sret});
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
};
} // namespace

//===----------------------------------------------------------------------===//
// x86_64 (x86 64 bit) linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetX86_64 : public GenericTarget<TargetX86_64> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;
};
} // namespace

//===----------------------------------------------------------------------===//
// AArch64 linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetAArch64 : public GenericTarget<TargetAArch64> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;
};
} // namespace

//===----------------------------------------------------------------------===//
// PPC64le linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetPPC64le : public GenericTarget<TargetPPC64le> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;
};
} // namespace

// Instantiate the overloaded target instance based on the triple value.
// Currently, the implementation only instantiates `i386-unknown-linux-gnu`,
// `x86_64-unknown-linux-gnu`, aarch64 and ppc64le like triples. Other targets
// should be added to this file as needed.
std::unique_ptr<fir::CodeGenSpecifics>
fir::CodeGenSpecifics::get(mlir::MLIRContext *ctx, llvm::Triple &&trp,
                           KindMapping &&kindMap) {
  switch (trp.getArch()) {
  default:
    break;
  case llvm::Triple::ArchType::x86:
    switch (trp.getOS()) {
    default:
      break;
    case llvm::Triple::OSType::Linux:
    case llvm::Triple::OSType::Darwin:
      return std::make_unique<TargetI386>(ctx, std::move(trp),
                                          std::move(kindMap));
    }
    break;
  case llvm::Triple::ArchType::x86_64:
    switch (trp.getOS()) {
    default:
      break;
    case llvm::Triple::OSType::Linux:
    case llvm::Triple::OSType::Darwin:
      return std::make_unique<TargetX86_64>(ctx, std::move(trp),
                                            std::move(kindMap));
    }
    break;
  case llvm::Triple::ArchType::aarch64:
    switch (trp.getOS()) {
    default:
      break;
    case llvm::Triple::OSType::Linux:
    case llvm::Triple::OSType::Darwin:
      return std::make_unique<TargetAArch64>(ctx, std::move(trp),
                                             std::move(kindMap));
    }
    break;
  case llvm::Triple::ArchType::ppc64le:
    switch (trp.getOS()) {
    default:
      break;
    case llvm::Triple::OSType::Linux:
      return std::make_unique<TargetPPC64le>(ctx, std::move(trp),
                                             std::move(kindMap));
    }
    break;
  }
  llvm::report_fatal_error("target not implemented");
}
