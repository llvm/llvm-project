//===-- Coarray.h -- generate Coarray intrinsics runtime calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H

#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

// Get the function type for a prif subroutine with a variable number of
// arguments
#define PRIF_FUNCTYPE(...)                                                     \
  mlir::FunctionType::get(builder.getContext(), /*inputs*/ {__VA_ARGS__},      \
                          /*result*/ {})

// Default prefix for subroutines of PRIF compiled with LLVM
#define PRIFNAME_SUB(fmt)                                                      \
  []() {                                                                       \
    std::ostringstream oss;                                                    \
    oss << "prif_" << fmt;                                                     \
    return fir::NameUniquer::doProcedure({"prif"}, {}, oss.str());             \
  }()

#define PRIF_STAT_TYPE builder.getRefType(builder.getI32Type())
#define PRIF_ERRMSG_TYPE                                                       \
  fir::BoxType::get(fir::CharacterType::get(builder.getContext(), 1,           \
                                            fir::CharacterType::unknownLen()))

/// Generate Call to runtime prif_init
mlir::Value genInitCoarray(fir::FirOpBuilder &builder, mlir::Location loc);

/// Generate Call to runtime prif_num_images
mlir::Value getNumImages(fir::FirOpBuilder &builder, mlir::Location loc);

/// Generate Call to runtime prif_num_images_with_team or
/// prif_num_images_with_team_number
mlir::Value getNumImagesWithTeam(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value team);

/// Generate Call to runtime prif_this_image_no_coarray
mlir::Value getThisImage(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value team = {});

/// Generate call to runtime subroutine prif_co_broadcast
void genCoBroadcast(fir::FirOpBuilder &builder, mlir::Location loc,
                    mlir::Value A, mlir::Value sourceImage, mlir::Value stat,
                    mlir::Value errmsg);

/// Generate call to runtime subroutine prif_co_max and prif_co_max_character
void genCoMax(fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value A,
              mlir::Value resultImage, mlir::Value stat, mlir::Value errmsg);

/// Generate call to runtime subroutine prif_co_min or prif_co_min_character
void genCoMin(fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value A,
              mlir::Value resultImage, mlir::Value stat, mlir::Value errmsg);

/// Generate call to runtime subroutine prif_co_sum
void genCoSum(fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value A,
              mlir::Value resultImage, mlir::Value stat, mlir::Value errmsg);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H
