//===-- Lower/Support/LoopAnnotation.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Support/LoopAnnotation.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/parse-tree.h"
#include "mlir/IR/BuiltinAttributes.h"

// For unroll directives without a value, force full unrolling.
// For unroll directives with a value, if the value is greater than 1,
// force unrolling with the given factor. Otherwise, disable unrolling.
static mlir::LLVM::LoopUnrollAttr
genLoopUnrollAttr(mlir::MLIRContext *context,
                  std::optional<std::uint64_t> directiveArg) {
  mlir::BoolAttr falseAttr = mlir::BoolAttr::get(context, false);
  mlir::BoolAttr trueAttr = mlir::BoolAttr::get(context, true);
  mlir::IntegerAttr countAttr;
  mlir::BoolAttr fullUnrollAttr;
  bool shouldUnroll = true;
  if (directiveArg.has_value()) {
    auto unrollingFactor = directiveArg.value();
    if (unrollingFactor == 0 || unrollingFactor == 1) {
      shouldUnroll = false;
    } else {
      countAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                         unrollingFactor);
    }
  } else {
    fullUnrollAttr = trueAttr;
  }

  mlir::BoolAttr disableAttr = shouldUnroll ? falseAttr : trueAttr;
  return mlir::LLVM::LoopUnrollAttr::get(context, /*disable=*/disableAttr,
                                         /*count=*/countAttr, {},
                                         /*full=*/fullUnrollAttr, {}, {}, {});
}

static mlir::LLVM::LoopUnrollAndJamAttr
genLoopUnrollAndJamAttr(mlir::MLIRContext *context,
                        std::optional<std::uint64_t> count) {
  mlir::BoolAttr falseAttr = mlir::BoolAttr::get(context, false);
  mlir::BoolAttr trueAttr = mlir::BoolAttr::get(context, true);
  mlir::IntegerAttr countAttr;
  bool shouldUnroll = true;
  if (count.has_value()) {
    auto unrollingFactor = count.value();
    if (unrollingFactor == 0 || unrollingFactor == 1) {
      shouldUnroll = false;
    } else {
      countAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                         unrollingFactor);
    }
  }

  mlir::BoolAttr disableAttr = shouldUnroll ? falseAttr : trueAttr;
  return mlir::LLVM::LoopUnrollAndJamAttr::get(context, /*disable=*/disableAttr,
                                               /*count*/ countAttr, {}, {}, {},
                                               {}, {});
}

static mlir::LLVM::LoopVectorizeAttr
genLoopVectorizeAttr(mlir::MLIRContext *context, mlir::BoolAttr disableAttr,
                     mlir::BoolAttr scalableEnable,
                     mlir::IntegerAttr vectorWidth) {
  mlir::LLVM::LoopVectorizeAttr va;
  if (disableAttr)
    va = mlir::LLVM::LoopVectorizeAttr::get(context,
                                            /*disable=*/disableAttr,
                                            /*predicate=*/{},
                                            /*scalableEnable=*/scalableEnable,
                                            /*vectorWidth=*/vectorWidth, {}, {},
                                            {});
  return va;
}

mlir::LLVM::LoopAnnotationAttr Fortran::lower::genLoopAnnotationAttr(
    mlir::MLIRContext *context,
    llvm::ArrayRef<const Fortran::parser::CompilerDirective *> dirs) {
  mlir::BoolAttr disableVecAttr;
  mlir::BoolAttr scalableEnable;
  mlir::IntegerAttr vectorWidth;
  mlir::LLVM::LoopUnrollAttr ua;
  mlir::LLVM::LoopUnrollAndJamAttr uja;
  llvm::SmallVector<mlir::LLVM::AccessGroupAttr> aga;
  bool hasAttrs = false;
  for (const auto *dir : dirs) {
    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::CompilerDirective::VectorAlways &) {
              disableVecAttr = mlir::BoolAttr::get(context, false);
              hasAttrs = true;
            },
            [&](const Fortran::parser::CompilerDirective::VectorLength &vl) {
              using Kind =
                  Fortran::parser::CompilerDirective::VectorLength::Kind;
              Kind kind = std::get<Kind>(vl.t);
              uint64_t length = std::get<uint64_t>(vl.t);
              disableVecAttr = mlir::BoolAttr::get(context, false);
              if (length != 0)
                vectorWidth = mlir::IntegerAttr::get(
                    mlir::IntegerType::get(context, 64), length);
              switch (kind) {
              case Kind::Scalable:
                scalableEnable = mlir::BoolAttr::get(context, true);
                break;
              case Kind::Fixed:
                scalableEnable = mlir::BoolAttr::get(context, false);
                break;
              case Kind::Auto:
                break;
              }
              hasAttrs = true;
            },
            [&](const Fortran::parser::CompilerDirective::Unroll &u) {
              ua = genLoopUnrollAttr(context, u.v);
              hasAttrs = true;
            },
            [&](const Fortran::parser::CompilerDirective::UnrollAndJam &u) {
              uja = genLoopUnrollAndJamAttr(context, u.v);
              hasAttrs = true;
            },
            [&](const Fortran::parser::CompilerDirective::NoVector &) {
              disableVecAttr = mlir::BoolAttr::get(context, true);
              hasAttrs = true;
            },
            [&](const Fortran::parser::CompilerDirective::NoUnroll &) {
              ua = genLoopUnrollAttr(context, /*directiveArg=*/0);
              hasAttrs = true;
            },
            [&](const Fortran::parser::CompilerDirective::NoUnrollAndJam &) {
              uja = genLoopUnrollAndJamAttr(context, /*count=*/0);
              hasAttrs = true;
            },
            [&](const Fortran::parser::CompilerDirective::IVDep &) {
              aga.push_back(mlir::LLVM::AccessGroupAttr::get(context));
              hasAttrs = true;
            },
            [&](const Fortran::parser::CompilerDirective::Simd &) {
              disableVecAttr = mlir::BoolAttr::get(context, false);
              hasAttrs = true;
            },
            [&](const auto &) {}},
        dir->u);
  }
  if (!hasAttrs)
    return {};

  mlir::LLVM::LoopVectorizeAttr va = genLoopVectorizeAttr(
      context, disableVecAttr, scalableEnable, vectorWidth);
  return mlir::LLVM::LoopAnnotationAttr::get(
      context, {}, /*vectorize=*/va, {}, /*unroll*/ ua,
      /*unroll_and_jam*/ uja, {}, {}, {}, {}, {}, {}, {}, {}, {},
      /*parallelAccesses*/ aga);
}
