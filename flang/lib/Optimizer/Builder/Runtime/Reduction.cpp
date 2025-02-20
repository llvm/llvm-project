//===-- Reduction.cpp -- generate reduction intrinsics runtime calls- -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Reduction.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Runtime/reduce.h"
#include "flang/Runtime/reduction.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace Fortran::runtime;

#define STRINGIFY(S) #S
#define JOIN2(A, B) A##B
#define JOIN3(A, B, C) A##B##C

/// Placeholder for real*10 version of Maxval Intrinsic
struct ForcedMaxvalReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MaxvalReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*16 version of Maxval Intrinsic
struct ForcedMaxvalReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MaxvalReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer*16 version of Maxval Intrinsic
struct ForcedMaxvalInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(MaxvalInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for unsigned*16 version of Maxval Intrinsic
struct ForcedMaxvalUnsigned16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(MaxvalUnsigned16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(
          ctx, 128, mlir::IntegerType::SignednessSemantics::Unsigned);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*10 version of Minval Intrinsic
struct ForcedMinvalReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MinvalReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*16 version of Minval Intrinsic
struct ForcedMinvalReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MinvalReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer*16 version of Minval Intrinsic
struct ForcedMinvalInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(MinvalInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for unsigned*16 version of Minval Intrinsic
struct ForcedMinvalUnsigned16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(MinvalUnsigned16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(
          ctx, 128, mlir::IntegerType::SignednessSemantics::Unsigned);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

// Maxloc/Minloc take descriptor, so these runtime signature are not ifdef
// and the mkRTKey can safely be used here. Define alias so that the
// REAL_INTRINSIC_INSTANCES macro works with them too
using ForcedMaxlocReal10 = mkRTKey(MaxlocReal10);
using ForcedMaxlocReal16 = mkRTKey(MaxlocReal16);
using ForcedMaxlocInteger16 = mkRTKey(MaxlocInteger16);
using ForcedMaxlocUnsigned16 = mkRTKey(MaxlocUnsigned16);
using ForcedMinlocReal10 = mkRTKey(MinlocReal10);
using ForcedMinlocReal16 = mkRTKey(MinlocReal16);
using ForcedMinlocInteger16 = mkRTKey(MinlocInteger16);
using ForcedMinlocUnsigned16 = mkRTKey(MinlocUnsigned16);

/// Placeholder for real*10 version of Norm2 Intrinsic
struct ForcedNorm2Real10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Norm2_10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Norm2 Intrinsic
struct ForcedNorm2Real16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Norm2_16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Norm2Dim Intrinsic
struct ForcedNorm2DimReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Norm2DimReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(
          ctx, {fir::ReferenceType::get(boxTy), boxTy, intTy, strTy, intTy},
          {});
    };
  }
};

/// Placeholder for real*10 version of Product Intrinsic
struct ForcedProductReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ProductReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*16 version of Product Intrinsic
struct ForcedProductReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ProductReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer*16 version of Product Intrinsic
struct ForcedProductInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ProductInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for unsigned*16 version of Product Intrinsic
struct ForcedProductUnsigned16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ProductUnsigned16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(
          ctx, 128, mlir::IntegerType::SignednessSemantics::Unsigned);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for complex(10) version of Product Intrinsic
struct ForcedProductComplex10 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppProductComplex10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float80Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(
          ctx, {resTy, boxTy, strTy, intTy, intTy, boxTy}, {});
    };
  }
};

/// Placeholder for complex(16) version of Product Intrinsic
struct ForcedProductComplex16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppProductComplex16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float128Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(
          ctx, {resTy, boxTy, strTy, intTy, intTy, boxTy}, {});
    };
  }
};

/// Placeholder for real*10 version of DotProduct Intrinsic
struct ForcedDotProductReal10 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(DotProductReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, boxTy, strTy, intTy}, {ty});
    };
  }
};

/// Placeholder for real*16 version of DotProduct Intrinsic
struct ForcedDotProductReal16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(DotProductReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, boxTy, strTy, intTy}, {ty});
    };
  }
};

/// Placeholder for complex(10) version of DotProduct Intrinsic
struct ForcedDotProductComplex10 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppDotProductComplex10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float80Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(ctx, {resTy, boxTy, boxTy, strTy, intTy},
                                     {});
    };
  }
};

/// Placeholder for complex(16) version of DotProduct Intrinsic
struct ForcedDotProductComplex16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppDotProductComplex16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float128Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(ctx, {resTy, boxTy, boxTy, strTy, intTy},
                                     {});
    };
  }
};

/// Placeholder for integer*16 version of DotProduct Intrinsic
struct ForcedDotProductInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(DotProductInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, boxTy, strTy, intTy}, {ty});
    };
  }
};

/// Placeholder for unsigned*16 version of DotProduct Intrinsic
struct ForcedDotProductUnsigned16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(DotProductUnsigned16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(
          ctx, 128, mlir::IntegerType::SignednessSemantics::Unsigned);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, boxTy, strTy, intTy}, {ty});
    };
  }
};

/// Placeholder for real*10 version of Sum Intrinsic
struct ForcedSumReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SumReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*16 version of Sum Intrinsic
struct ForcedSumReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SumReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer*16 version of Sum Intrinsic
struct ForcedSumInteger16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SumInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for unsigned*16 version of Sum Intrinsic
struct ForcedSumUnsigned16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SumUnsigned16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(
          ctx, 128, mlir::IntegerType::SignednessSemantics::Unsigned);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for complex(10) version of Sum Intrinsic
struct ForcedSumComplex10 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppSumComplex10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float80Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(
          ctx, {resTy, boxTy, strTy, intTy, intTy, boxTy}, {});
    };
  }
};

/// Placeholder for complex(16) version of Sum Intrinsic
struct ForcedSumComplex16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppSumComplex16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float128Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(
          ctx, {resTy, boxTy, strTy, intTy, intTy, boxTy}, {});
    };
  }
};

/// Placeholder for integer(16) version of IAll Intrinsic
struct ForcedIAll16 {
  static constexpr const char *name = EXPAND_AND_QUOTE_KEY(IAll16);
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer(16) version of IAny Intrinsic
struct ForcedIAny16 {
  static constexpr const char *name = EXPAND_AND_QUOTE_KEY(IAny16);
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer(16) version of IParity Intrinsic
struct ForcedIParity16 {
  static constexpr const char *name = EXPAND_AND_QUOTE_KEY(IParity16);
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*10 version of Reduce Intrinsic
struct ForcedReduceReal10Ref {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceReal10Ref));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty}, {ty});
    };
  }
};

/// Placeholder for real*10 version of Reduce Intrinsic
struct ForcedReduceReal10Value {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceReal10Value));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Reduce Intrinsic
struct ForcedReduceReal16Ref {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceReal16Ref));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Reduce Intrinsic
struct ForcedReduceReal16Value {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceReal16Value));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty}, {ty});
    };
  }
};

/// Placeholder for DIM real*10 version of Reduce Intrinsic
struct ForcedReduceReal10DimRef {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceReal10DimRef));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for DIM real*10 with value version of Reduce Intrinsic
struct ForcedReduceReal10DimValue {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceReal10DimValue));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float80Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for DIM real*16 version of Reduce Intrinsic
struct ForcedReduceReal16DimRef {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceReal16DimRef));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for DIM real*16 with value version of Reduce Intrinsic
struct ForcedReduceReal16DimValue {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceReal16DimValue));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::Float128Type::get(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for integer*16 version of Reduce Intrinsic
struct ForcedReduceInteger16Ref {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceInteger16Ref));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty}, {ty});
    };
  }
};

/// Placeholder for unsigned*16 version of Reduce Intrinsic
struct ForcedReduceUnsigned16Ref {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceUnsigned16Ref));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty}, {ty});
    };
  }
};

/// Placeholder for integer*16 with value version of Reduce Intrinsic
struct ForcedReduceInteger16Value {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceInteger16Value));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty}, {ty});
    };
  }
};

/// Placeholder for unsigned*16 with value version of Reduce Intrinsic
struct ForcedReduceUnsigned16Value {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceUnsigned16Value));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty}, {ty});
    };
  }
};

/// Placeholder for DIM integer*16 version of Reduce Intrinsic
struct ForcedReduceInteger16DimRef {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceInteger16DimRef));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for DIM unsigned*16 version of Reduce Intrinsic
struct ForcedReduceUnsigned16DimRef {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceUnsigned16DimRef));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(
          ctx, 128, mlir::IntegerType::SignednessSemantics::Unsigned);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for DIM integer*16 with value version of Reduce Intrinsic
struct ForcedReduceInteger16DimValue {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceInteger16DimValue));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for DIM unsigned*16 with value version of Reduce Intrinsic
struct ForcedReduceUnsigned16DimValue {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ReduceUnsigned16DimValue));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(
          ctx, 128, mlir::IntegerType::SignednessSemantics::Unsigned);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for complex(10) version of Reduce Intrinsic
struct ForcedReduceComplex10Ref {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppReduceComplex10Ref));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float80Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for complex(10) with value version of Reduce Intrinsic
struct ForcedReduceComplex10Value {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppReduceComplex10Value));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float80Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for Dim complex(10) version of Reduce Intrinsic
struct ForcedReduceComplex10DimRef {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppReduceComplex10DimRef));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float80Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for Dim complex(10) with value version of Reduce Intrinsic
struct ForcedReduceComplex10DimValue {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppReduceComplex10DimValue));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float80Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for complex(16) version of Reduce Intrinsic
struct ForcedReduceComplex16Ref {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppReduceComplex16Ref));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float128Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for complex(16) with value version of Reduce Intrinsic
struct ForcedReduceComplex16Value {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppReduceComplex16Value));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float128Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for Dim complex(16) version of Reduce Intrinsic
struct ForcedReduceComplex16DimRef {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppReduceComplex16DimRef));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float128Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {refTy, refTy}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

/// Placeholder for Dim complex(16) with value version of Reduce Intrinsic
struct ForcedReduceComplex16DimValue {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppReduceComplex16DimValue));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::Float128Type::get(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto refTy = fir::ReferenceType::get(ty);
      auto opTy = mlir::FunctionType::get(ctx, {ty, ty}, refTy);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto refBoxTy = fir::ReferenceType::get(boxTy);
      auto i1Ty = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(
          ctx, {refBoxTy, boxTy, opTy, strTy, intTy, intTy, boxTy, refTy, i1Ty},
          {});
    };
  }
};

#define INTRINSIC_INSTANCE(NAME, CAT, KIND, SUFFIX)                            \
  if (!func && cat == TypeCategory::CAT && kind == KIND) {                     \
    func = fir::runtime::getRuntimeFunc<mkRTKey(NAME##CAT##KIND##SUFFIX)>(     \
        loc, builder);                                                         \
  }
#define FORCED_INTRINSIC_INSTANCE(NAME, CAT, KIND, SUFFIX)                     \
  if (!func && cat == TypeCategory::CAT && kind == KIND) {                     \
    func = fir::runtime::getRuntimeFunc<Forced##NAME##CAT##KIND##SUFFIX>(      \
        loc, builder);                                                         \
  }

#define INTEGER_INTRINSIC_INSTANCES(NAME, SUFFIX)                              \
  INTRINSIC_INSTANCE(NAME, Integer, 1, SUFFIX)                                 \
  INTRINSIC_INSTANCE(NAME, Integer, 2, SUFFIX)                                 \
  INTRINSIC_INSTANCE(NAME, Integer, 4, SUFFIX)                                 \
  INTRINSIC_INSTANCE(NAME, Integer, 8, SUFFIX)                                 \
  FORCED_INTRINSIC_INSTANCE(NAME, Integer, 16, SUFFIX)

#define UNSIGNED_INTRINSIC_INSTANCES(NAME, SUFFIX)                             \
  INTRINSIC_INSTANCE(NAME, Unsigned, 1, SUFFIX)                                \
  INTRINSIC_INSTANCE(NAME, Unsigned, 2, SUFFIX)                                \
  INTRINSIC_INSTANCE(NAME, Unsigned, 4, SUFFIX)                                \
  INTRINSIC_INSTANCE(NAME, Unsigned, 8, SUFFIX)                                \
  FORCED_INTRINSIC_INSTANCE(NAME, Unsigned, 16, SUFFIX)

#define REAL_INTRINSIC_INSTANCES(NAME, SUFFIX)                                 \
  INTRINSIC_INSTANCE(NAME, Real, 4, SUFFIX)                                    \
  INTRINSIC_INSTANCE(NAME, Real, 8, SUFFIX)                                    \
  FORCED_INTRINSIC_INSTANCE(NAME, Real, 10, SUFFIX)                            \
  FORCED_INTRINSIC_INSTANCE(NAME, Real, 16, SUFFIX)

#define COMPLEX_INTRINSIC_INSTANCES(NAME, SUFFIX)                              \
  INTRINSIC_INSTANCE(Cpp##NAME, Complex, 4, SUFFIX)                            \
  INTRINSIC_INSTANCE(Cpp##NAME, Complex, 8, SUFFIX)                            \
  FORCED_INTRINSIC_INSTANCE(NAME, Complex, 10, SUFFIX)                         \
  FORCED_INTRINSIC_INSTANCE(NAME, Complex, 16, SUFFIX)

#define NUMERICAL_INTRINSIC_INSTANCES(NAME)                                    \
  INTEGER_INTRINSIC_INSTANCES(NAME, )                                          \
  UNSIGNED_INTRINSIC_INSTANCES(NAME, )                                         \
  REAL_INTRINSIC_INSTANCES(NAME, )                                             \
  COMPLEX_INTRINSIC_INSTANCES(NAME, )

#define LOGICAL_INTRINSIC_INSTANCES(NAME, SUFFIX)                              \
  INTRINSIC_INSTANCE(NAME, Logical, 1, SUFFIX)                                 \
  INTRINSIC_INSTANCE(NAME, Logical, 2, SUFFIX)                                 \
  INTRINSIC_INSTANCE(NAME, Logical, 4, SUFFIX)                                 \
  INTRINSIC_INSTANCE(NAME, Logical, 8, SUFFIX)

#define NUMERICAL_AND_LOGICAL_INSTANCES(NAME, SUFFIX)                          \
  INTEGER_INTRINSIC_INSTANCES(NAME, SUFFIX)                                    \
  UNSIGNED_INTRINSIC_INSTANCES(NAME, SUFFIX)                                   \
  REAL_INTRINSIC_INSTANCES(NAME, SUFFIX)                                       \
  COMPLEX_INTRINSIC_INSTANCES(NAME, SUFFIX)                                    \
  LOGICAL_INTRINSIC_INSTANCES(NAME, SUFFIX)

// REAL/COMPLEX 2 and 3 usually have no runtime implementation, so they have
// special macros.
#define REAL_2_3_INTRINSIC_INSTANCES(NAME, SUFFIX)                             \
  INTRINSIC_INSTANCE(NAME, Real, 2, SUFFIX)                                    \
  INTRINSIC_INSTANCE(NAME, Real, 3, SUFFIX)

#define COMPLEX_2_3_INTRINSIC_INSTANCES(NAME, SUFFIX)                          \
  INTRINSIC_INSTANCE(Cpp##NAME, Complex, 2, SUFFIX)                            \
  INTRINSIC_INSTANCE(Cpp##NAME, Complex, 3, SUFFIX)

/// Generate call to specialized runtime function that takes a mask and
/// dim argument. The All, Any, and Count intrinsics use this pattern.
template <typename FN>
mlir::Value genSpecial2Args(FN func, fir::FirOpBuilder &builder,
                            mlir::Location loc, mlir::Value maskBox,
                            mlir::Value dim) {
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, maskBox,
                                            sourceFile, sourceLine, dim);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate calls to reduction intrinsics such as All and Any.
/// These are the descriptor based implementations that take two
/// arguments (mask, dim).
template <typename FN>
static void genReduction2Args(FN func, fir::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value maskBox, mlir::Value dim) {
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, maskBox, dim, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate calls to reduction intrinsics such as Maxval and Minval.
/// These take arguments such as (array, dim, mask).
template <typename FN>
static void genReduction3Args(FN func, fir::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value arrayBox, mlir::Value dim,
                              mlir::Value maskBox) {

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, arrayBox, dim,
                                    sourceFile, sourceLine, maskBox);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate calls to reduction intrinsics such as Maxloc and Minloc.
/// These take arguments such as (array, mask, kind, back).
template <typename FN>
static void genReduction4Args(FN func, fir::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value arrayBox, mlir::Value maskBox,
                              mlir::Value kind, mlir::Value back) {
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            arrayBox, kind, sourceFile,
                                            sourceLine, maskBox, back);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate calls to reduction intrinsics such as Maxloc and Minloc.
/// These take arguments such as (array, dim, mask, kind, back).
template <typename FN>
static void
genReduction5Args(FN func, fir::FirOpBuilder &builder, mlir::Location loc,
                  mlir::Value resultBox, mlir::Value arrayBox, mlir::Value dim,
                  mlir::Value maskBox, mlir::Value kind, mlir::Value back) {
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            arrayBox, kind, dim, sourceFile,
                                            sourceLine, maskBox, back);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `AllDim` runtime routine.
/// This calls the descriptor based runtime call implementation of the `all`
/// intrinsic.
void fir::runtime::genAllDescriptor(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value resultBox,
                                    mlir::Value maskBox, mlir::Value dim) {
  auto allFunc = fir::runtime::getRuntimeFunc<mkRTKey(AllDim)>(loc, builder);
  genReduction2Args(allFunc, builder, loc, resultBox, maskBox, dim);
}

/// Generate call to `AnyDim` runtime routine.
/// This calls the descriptor based runtime call implementation of the `any`
/// intrinsic.
void fir::runtime::genAnyDescriptor(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value resultBox,
                                    mlir::Value maskBox, mlir::Value dim) {
  auto anyFunc = fir::runtime::getRuntimeFunc<mkRTKey(AnyDim)>(loc, builder);
  genReduction2Args(anyFunc, builder, loc, resultBox, maskBox, dim);
}

/// Generate call to `ParityDim` runtime routine.
/// This calls the descriptor based runtime call implementation of the `parity`
/// intrinsic.
void fir::runtime::genParityDescriptor(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value resultBox,
                                       mlir::Value maskBox, mlir::Value dim) {
  auto parityFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ParityDim)>(loc, builder);
  genReduction2Args(parityFunc, builder, loc, resultBox, maskBox, dim);
}

/// Generate call to `All` intrinsic runtime routine. This routine is
/// specialized for mask arguments with rank == 1.
mlir::Value fir::runtime::genAll(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value maskBox, mlir::Value dim) {
  auto allFunc = fir::runtime::getRuntimeFunc<mkRTKey(All)>(loc, builder);
  return genSpecial2Args(allFunc, builder, loc, maskBox, dim);
}

/// Generate call to `Any` intrinsic runtime routine. This routine is
/// specialized for mask arguments with rank == 1.
mlir::Value fir::runtime::genAny(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value maskBox, mlir::Value dim) {
  auto anyFunc = fir::runtime::getRuntimeFunc<mkRTKey(Any)>(loc, builder);
  return genSpecial2Args(anyFunc, builder, loc, maskBox, dim);
}

/// Generate call to `Count` runtime routine. This routine is a specialized
/// version when mask is a rank one array or the dim argument is not
/// specified by the user.
mlir::Value fir::runtime::genCount(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value maskBox,
                                   mlir::Value dim) {
  auto countFunc = fir::runtime::getRuntimeFunc<mkRTKey(Count)>(loc, builder);
  return genSpecial2Args(countFunc, builder, loc, maskBox, dim);
}

/// Generate call to general `CountDim` runtime routine. This routine has a
/// descriptor result.
void fir::runtime::genCountDim(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value resultBox, mlir::Value maskBox,
                               mlir::Value dim, mlir::Value kind) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(CountDim)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, maskBox, dim, kind, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `Findloc` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
void fir::runtime::genFindloc(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value resultBox, mlir::Value arrayBox,
                              mlir::Value valBox, mlir::Value maskBox,
                              mlir::Value kind, mlir::Value back) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Findloc)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            arrayBox, valBox, kind, sourceFile,
                                            sourceLine, maskBox, back);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `FindlocDim` intrinsic runtime routine. This is the version
/// that takes a dim argument.
void fir::runtime::genFindlocDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value arrayBox,
                                 mlir::Value valBox, mlir::Value dim,
                                 mlir::Value maskBox, mlir::Value kind,
                                 mlir::Value back) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(FindlocDim)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(6));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, arrayBox, valBox, kind, dim, sourceFile,
      sourceLine, maskBox, back);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `Maxloc` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
void fir::runtime::genMaxloc(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value arrayBox,
                             mlir::Value maskBox, mlir::Value kindVal,
                             mlir::Value back) {
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  fir::factory::CharacterExprHelper charHelper{builder, loc};
  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);
  mlir::func::FuncOp func;
  REAL_INTRINSIC_INSTANCES(Maxloc, )
  INTEGER_INTRINSIC_INSTANCES(Maxloc, )
  UNSIGNED_INTRINSIC_INSTANCES(Maxloc, )
  if (charHelper.isCharacterScalar(eleTy))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MaxlocCharacter)>(loc, builder);
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "MAXLOC");
  genReduction4Args(func, builder, loc, resultBox, arrayBox, maskBox, kindVal,
                    back);
}

/// Generate call to `MaxlocDim` intrinsic runtime routine. This is the version
/// that takes a dim argument.
void fir::runtime::genMaxlocDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value resultBox, mlir::Value arrayBox,
                                mlir::Value dim, mlir::Value maskBox,
                                mlir::Value kind, mlir::Value back) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(MaxlocDim)>(loc, builder);
  genReduction5Args(func, builder, loc, resultBox, arrayBox, dim, maskBox, kind,
                    back);
}

/// Generate call to `Maxval` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value fir::runtime::genMaxval(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value arrayBox,
                                    mlir::Value maskBox) {
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);
  mlir::func::FuncOp func;
  REAL_INTRINSIC_INSTANCES(Maxval, )
  INTEGER_INTRINSIC_INSTANCES(Maxval, )
  UNSIGNED_INTRINSIC_INSTANCES(Maxval, )
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "MAXVAL");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to `MaxvalDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genMaxvalDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value resultBox, mlir::Value arrayBox,
                                mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(MaxvalDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `MaxvalCharacter` intrinsic runtime routine. This is the
/// version that handles character arrays of rank 1 and without a DIM argument.
void fir::runtime::genMaxvalChar(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value arrayBox,
                                 mlir::Value maskBox) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(MaxvalCharacter)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, arrayBox, sourceFile, sourceLine, maskBox);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `Minloc` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
void fir::runtime::genMinloc(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value arrayBox,
                             mlir::Value maskBox, mlir::Value kindVal,
                             mlir::Value back) {
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);
  mlir::func::FuncOp func;
  REAL_INTRINSIC_INSTANCES(Minloc, )
  INTEGER_INTRINSIC_INSTANCES(Minloc, )
  UNSIGNED_INTRINSIC_INSTANCES(Minloc, )
  fir::factory::CharacterExprHelper charHelper{builder, loc};
  if (charHelper.isCharacterScalar(eleTy))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MinlocCharacter)>(loc, builder);
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "MINLOC");
  genReduction4Args(func, builder, loc, resultBox, arrayBox, maskBox, kindVal,
                    back);
}

/// Generate call to `MinlocDim` intrinsic runtime routine. This is the version
/// that takes a dim argument.
void fir::runtime::genMinlocDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value resultBox, mlir::Value arrayBox,
                                mlir::Value dim, mlir::Value maskBox,
                                mlir::Value kind, mlir::Value back) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(MinlocDim)>(loc, builder);
  genReduction5Args(func, builder, loc, resultBox, arrayBox, dim, maskBox, kind,
                    back);
}

/// Generate call to `MinvalDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genMinvalDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value resultBox, mlir::Value arrayBox,
                                mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(MinvalDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `MinvalCharacter` intrinsic runtime routine. This is the
/// version that handles character arrays of rank 1 and without a DIM argument.
void fir::runtime::genMinvalChar(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value arrayBox,
                                 mlir::Value maskBox) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(MinvalCharacter)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, arrayBox, sourceFile, sourceLine, maskBox);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `Minval` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value fir::runtime::genMinval(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value arrayBox,
                                    mlir::Value maskBox) {
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);

  mlir::func::FuncOp func;
  REAL_INTRINSIC_INSTANCES(Minval, )
  INTEGER_INTRINSIC_INSTANCES(Minval, )
  UNSIGNED_INTRINSIC_INSTANCES(Minval, )
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "MINVAL");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to `Norm2Dim` intrinsic runtime routine. This is the version
/// that takes a dim argument.
void fir::runtime::genNorm2Dim(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value resultBox, mlir::Value arrayBox,
                               mlir::Value dim) {
  mlir::func::FuncOp func;
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  if (eleTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedNorm2DimReal16>(loc, builder);
  else
    func = fir::runtime::getRuntimeFunc<mkRTKey(Norm2Dim)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, arrayBox, dim, sourceFile, sourceLine);

  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `Norm2` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value fir::runtime::genNorm2(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value arrayBox) {
  mlir::func::FuncOp func;
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);

  if (eleTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Norm2_4)>(loc, builder);
  else if (eleTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Norm2_8)>(loc, builder);
  else if (eleTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedNorm2Real10>(loc, builder);
  else if (eleTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedNorm2Real16>(loc, builder);
  else
    fir::intrinsicTypeTODO(builder, eleTy, loc, "NORM2");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, arrayBox,
                                            sourceFile, sourceLine, dim);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to `Parity` intrinsic runtime routine. This routine is
/// specialized for mask arguments with rank == 1.
mlir::Value fir::runtime::genParity(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value maskBox,
                                    mlir::Value dim) {
  auto parityFunc = fir::runtime::getRuntimeFunc<mkRTKey(Parity)>(loc, builder);
  return genSpecial2Args(parityFunc, builder, loc, maskBox, dim);
}

/// Generate call to `ProductDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genProductDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value arrayBox,
                                 mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(ProductDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `Product` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value fir::runtime::genProduct(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value arrayBox,
                                     mlir::Value maskBox,
                                     mlir::Value resultBox) {
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);

  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);
  mlir::func::FuncOp func;
  NUMERICAL_INTRINSIC_INSTANCES(Product)
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "PRODUCT");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  if (fir::isa_complex(eleTy)) {
    auto sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
    auto args =
        fir::runtime::createArguments(builder, loc, fTy, resultBox, arrayBox,
                                      sourceFile, sourceLine, dim, maskBox);
    builder.create<fir::CallOp>(loc, func, args);
    return resultBox;
  }

  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to `DotProduct` intrinsic runtime routine.
mlir::Value fir::runtime::genDotProduct(fir::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value vectorABox,
                                        mlir::Value vectorBBox,
                                        mlir::Value resultBox) {
  // For complex data types, resultBox is !fir.ref<!fir.complex<N>>,
  // otherwise it is !fir.box<T>.
  auto ty = resultBox.getType();
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(ty);

  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);
  mlir::func::FuncOp func;
  NUMERICAL_INTRINSIC_INSTANCES(DotProduct)
  if (cat == Fortran::common::TypeCategory::Logical)
    func =
        fir::runtime::getRuntimeFunc<mkRTKey(DotProductLogical)>(loc, builder);
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "DOTPRODUCT");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);

  if (fir::isa_complex(eleTy)) {
    auto sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
    auto args =
        fir::runtime::createArguments(builder, loc, fTy, resultBox, vectorABox,
                                      vectorBBox, sourceFile, sourceLine);
    builder.create<fir::CallOp>(loc, func, args);
    return resultBox;
  }

  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, vectorABox,
                                            vectorBBox, sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
/// Generate call to `SumDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genSumDim(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value arrayBox,
                             mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(SumDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `Sum` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value fir::runtime::genSum(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value arrayBox, mlir::Value maskBox,
                                 mlir::Value resultBox) {
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);

  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);
  mlir::func::FuncOp func;
  NUMERICAL_INTRINSIC_INSTANCES(Sum)
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "SUM");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  if (fir::isa_complex(eleTy)) {
    auto sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
    auto args =
        fir::runtime::createArguments(builder, loc, fTy, resultBox, arrayBox,
                                      sourceFile, sourceLine, dim, maskBox);
    builder.create<fir::CallOp>(loc, func, args);
    return resultBox;
  }

  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

// The IAll, IAny and IParity intrinsics have essentially the same
// implementation. This macro will generate the function body given the
// intrinsic name.
#define GEN_IALL_IANY_IPARITY(F)                                               \
  mlir::Value fir::runtime::JOIN2(gen, F)(                                     \
      fir::FirOpBuilder & builder, mlir::Location loc, mlir::Value arrayBox,   \
      mlir::Value maskBox, mlir::Value resultBox) {                            \
    mlir::func::FuncOp func;                                                   \
    auto ty = arrayBox.getType();                                              \
    auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);                              \
    auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();        \
    auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);  \
                                                                               \
    if (eleTy.isInteger(builder.getKindMap().getIntegerBitsize(1)))            \
      func = fir::runtime::getRuntimeFunc<mkRTKey(JOIN2(F, 1))>(loc, builder); \
    else if (eleTy.isInteger(builder.getKindMap().getIntegerBitsize(2)))       \
      func = fir::runtime::getRuntimeFunc<mkRTKey(JOIN2(F, 2))>(loc, builder); \
    else if (eleTy.isInteger(builder.getKindMap().getIntegerBitsize(4)))       \
      func = fir::runtime::getRuntimeFunc<mkRTKey(JOIN2(F, 4))>(loc, builder); \
    else if (eleTy.isInteger(builder.getKindMap().getIntegerBitsize(8)))       \
      func = fir::runtime::getRuntimeFunc<mkRTKey(JOIN2(F, 8))>(loc, builder); \
    else if (eleTy.isInteger(builder.getKindMap().getIntegerBitsize(16)))      \
      func = fir::runtime::getRuntimeFunc<JOIN3(Forced, F, 16)>(loc, builder); \
    else                                                                       \
      fir::emitFatalError(loc, "invalid type in " STRINGIFY(F));               \
                                                                               \
    auto fTy = func.getFunctionType();                                         \
    auto sourceFile = fir::factory::locationToFilename(builder, loc);          \
    auto sourceLine =                                                          \
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));         \
    auto args = fir::runtime::createArguments(                                 \
        builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);    \
                                                                               \
    return builder.create<fir::CallOp>(loc, func, args).getResult(0);          \
  }

/// Generate call to `IAllDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genIAllDim(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value resultBox, mlir::Value arrayBox,
                              mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(IAllDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `IAll` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
GEN_IALL_IANY_IPARITY(IAll)

/// Generate call to `IAnyDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genIAnyDim(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value resultBox, mlir::Value arrayBox,
                              mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(IAnyDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `IAny` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
GEN_IALL_IANY_IPARITY(IAny)

/// Generate call to `IParityDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genIParityDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value arrayBox,
                                 mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(IParityDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `IParity` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
GEN_IALL_IANY_IPARITY(IParity)

/// Generate call to `Reduce` intrinsic runtime routine. This is the version
/// that does not take a DIM argument and store result in the passed result
/// value.
void fir::runtime::genReduce(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value arrayBox, mlir::Value operation,
                             mlir::Value maskBox, mlir::Value identity,
                             mlir::Value ordered, mlir::Value resultBox,
                             bool argByRef) {
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  auto dim = builder.createIntegerConstant(loc, builder.getI32Type(), 1);

  assert(resultBox && "expect non null value for the result");
  assert((fir::isa_char(eleTy) || fir::isa_complex(eleTy) ||
          fir::isa_derived(eleTy)) &&
         "expect character, complex or derived-type");

  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);
  mlir::func::FuncOp func;
  if (argByRef) {
    COMPLEX_2_3_INTRINSIC_INSTANCES(Reduce, Ref)
    COMPLEX_INTRINSIC_INSTANCES(Reduce, Ref)
  } else {
    COMPLEX_2_3_INTRINSIC_INSTANCES(Reduce, Value)
    COMPLEX_INTRINSIC_INSTANCES(Reduce, Value)
  }
  fir::factory::CharacterExprHelper charHelper{builder, loc};
  if (fir::isa_char(eleTy) && charHelper.getCharacterKind(eleTy) == 1)
    func = fir::runtime::getRuntimeFunc<mkRTKey(ReduceChar1)>(loc, builder);
  else if (fir::isa_char(eleTy) && charHelper.getCharacterKind(eleTy) == 2)
    func = fir::runtime::getRuntimeFunc<mkRTKey(ReduceChar2)>(loc, builder);
  else if (fir::isa_char(eleTy) && charHelper.getCharacterKind(eleTy) == 4)
    func = fir::runtime::getRuntimeFunc<mkRTKey(ReduceChar4)>(loc, builder);
  else if (fir::isa_derived(eleTy))
    func =
        fir::runtime::getRuntimeFunc<mkRTKey(ReduceDerivedType)>(loc, builder);
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "REDUCE");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto opAddr = builder.create<fir::BoxAddrOp>(loc, fTy.getInput(2), operation);
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, arrayBox, opAddr, sourceFile, sourceLine,
      dim, maskBox, identity, ordered);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `Reduce` intrinsic runtime routine. This is the version
/// that does not take DIM argument and return a scalar result.
mlir::Value fir::runtime::genReduce(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value arrayBox,
                                    mlir::Value operation, mlir::Value maskBox,
                                    mlir::Value identity, mlir::Value ordered,
                                    bool argByRef) {
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  auto dim = builder.createIntegerConstant(loc, builder.getI32Type(), 1);

  assert((fir::isa_real(eleTy) || fir::isa_integer(eleTy) ||
          mlir::isa<fir::LogicalType>(eleTy)) &&
         "expect real, interger or logical");

  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);
  mlir::func::FuncOp func;
  if (argByRef) {
    REAL_2_3_INTRINSIC_INSTANCES(Reduce, Ref)
    REAL_INTRINSIC_INSTANCES(Reduce, Ref)
    INTEGER_INTRINSIC_INSTANCES(Reduce, Ref)
    UNSIGNED_INTRINSIC_INSTANCES(Reduce, Ref)
    LOGICAL_INTRINSIC_INSTANCES(Reduce, Ref)
  } else {
    REAL_2_3_INTRINSIC_INSTANCES(Reduce, Value)
    REAL_INTRINSIC_INSTANCES(Reduce, Value)
    INTEGER_INTRINSIC_INSTANCES(Reduce, Value)
    UNSIGNED_INTRINSIC_INSTANCES(Reduce, Value)
    LOGICAL_INTRINSIC_INSTANCES(Reduce, Value)
  }
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "REDUCE");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto opAddr = builder.create<fir::BoxAddrOp>(loc, fTy.getInput(1), operation);
  auto args = fir::runtime::createArguments(builder, loc, fTy, arrayBox, opAddr,
                                            sourceFile, sourceLine, dim,
                                            maskBox, identity, ordered);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

void fir::runtime::genReduceDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value arrayBox, mlir::Value operation,
                                mlir::Value dim, mlir::Value maskBox,
                                mlir::Value identity, mlir::Value ordered,
                                mlir::Value resultBox, bool argByRef) {
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = mlir::cast<fir::SequenceType>(arrTy).getElementType();
  auto [cat, kind] = fir::mlirTypeToCategoryKind(loc, eleTy);

  mlir::func::FuncOp func;
  if (argByRef) {
    REAL_2_3_INTRINSIC_INSTANCES(Reduce, DimRef)
    COMPLEX_2_3_INTRINSIC_INSTANCES(Reduce, DimRef)
    NUMERICAL_AND_LOGICAL_INSTANCES(Reduce, DimRef)
  } else {
    REAL_2_3_INTRINSIC_INSTANCES(Reduce, DimValue)
    COMPLEX_2_3_INTRINSIC_INSTANCES(Reduce, DimValue)
    NUMERICAL_AND_LOGICAL_INSTANCES(Reduce, DimValue)
  }
  fir::factory::CharacterExprHelper charHelper{builder, loc};
  if (fir::isa_char(eleTy) && charHelper.getCharacterKind(eleTy) == 1)
    func = fir::runtime::getRuntimeFunc<mkRTKey(ReduceCharacter1Dim)>(loc,
                                                                      builder);
  else if (fir::isa_char(eleTy) && charHelper.getCharacterKind(eleTy) == 2)
    func = fir::runtime::getRuntimeFunc<mkRTKey(ReduceCharacter2Dim)>(loc,
                                                                      builder);
  else if (fir::isa_char(eleTy) && charHelper.getCharacterKind(eleTy) == 4)
    func = fir::runtime::getRuntimeFunc<mkRTKey(ReduceCharacter4Dim)>(loc,
                                                                      builder);
  else if (fir::isa_derived(eleTy))
    func = fir::runtime::getRuntimeFunc<mkRTKey(ReduceDerivedTypeDim)>(loc,
                                                                       builder);
  if (!func)
    fir::intrinsicTypeTODO(builder, eleTy, loc, "REDUCE");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);

  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto opAddr = builder.create<fir::BoxAddrOp>(loc, fTy.getInput(2), operation);
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, arrayBox, opAddr, sourceFile, sourceLine,
      dim, maskBox, identity, ordered);
  builder.create<fir::CallOp>(loc, func, args);
}
