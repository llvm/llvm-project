//===-- DescriptorModel.h -- model of descriptors for codegen ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR dialect models of C++ types.
//
// This supplies a set of model builders to decompose the C declaration of a
// descriptor (as encoded in ISO_Fortran_binding.h and elsewhere) and
// reconstruct that type in the LLVM IR dialect.
//
// TODO: It is understood that this is deeply incorrect as far as building a
// portability layer for cross-compilation as these reflected types are those of
// the build machine and not necessarily that of either the host or the target.
// This assumption that build == host == target is actually pervasive across the
// compiler (https://llvm.org/PR52418).
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_DESCRIPTOR_MODEL_H
#define OPTIMIZER_DESCRIPTOR_MODEL_H

#include "flang/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/descriptor.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <tuple>

namespace fir {

using TypeBuilderFunc = mlir::Type (*)(mlir::MLIRContext *);

/// Get the LLVM IR dialect model for building a particular C++ type, `T`.
template <typename T>
static TypeBuilderFunc getModel();

template <>
constexpr TypeBuilderFunc getModel<void *>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::LLVM::LLVMPointerType::get(context);
  };
}
template <>
constexpr TypeBuilderFunc getModel<unsigned>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, sizeof(unsigned) * 8);
  };
}
template <>
constexpr TypeBuilderFunc getModel<int>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, sizeof(int) * 8);
  };
}
template <>
constexpr TypeBuilderFunc getModel<unsigned long>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, sizeof(unsigned long) * 8);
  };
}
template <>
constexpr TypeBuilderFunc getModel<unsigned long long>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, sizeof(unsigned long long) * 8);
  };
}
template <>
constexpr TypeBuilderFunc getModel<long long>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, sizeof(long long) * 8);
  };
}
template <>
constexpr TypeBuilderFunc getModel<Fortran::ISO::CFI_rank_t>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context,
                                  sizeof(Fortran::ISO::CFI_rank_t) * 8);
  };
}
template <>
constexpr TypeBuilderFunc getModel<Fortran::ISO::CFI_type_t>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context,
                                  sizeof(Fortran::ISO::CFI_type_t) * 8);
  };
}
template <>
constexpr TypeBuilderFunc getModel<long>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, sizeof(long) * 8);
  };
}
template <>
constexpr TypeBuilderFunc getModel<Fortran::ISO::CFI_dim_t>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    auto indexTy = getModel<Fortran::ISO::CFI_index_t>()(context);
    return mlir::LLVM::LLVMArrayType::get(indexTy, 3);
  };
}
template <>
constexpr TypeBuilderFunc
getModel<Fortran::ISO::cfi_internal::FlexibleArray<Fortran::ISO::CFI_dim_t>>() {
  return getModel<Fortran::ISO::CFI_dim_t>();
}

//===----------------------------------------------------------------------===//
// Descriptor reflection
//===----------------------------------------------------------------------===//

/// Get the type model of the field number `Field` in an ISO CFI descriptor.
template <int Field>
static constexpr TypeBuilderFunc getDescFieldTypeModel() {
  Fortran::ISO::CFI_cdesc_t dummyDesc{};
  // check that the descriptor is exactly 8 fields as specified in CFI_cdesc_t
  // in flang/include/flang/ISO_Fortran_binding.h.
  auto [a, b, c, d, e, f, g, h] = dummyDesc;
  auto tup = std::tie(a, b, c, d, e, f, g, h);
  auto field = std::get<Field>(tup);
  return getModel<decltype(field)>();
}

/// An extended descriptor is defined by a class in runtime/descriptor.h. The
/// three fields in the class are hard-coded here, unlike the reflection used on
/// the ISO parts, which are a POD.
template <int Field>
static constexpr TypeBuilderFunc getExtendedDescFieldTypeModel() {
  if constexpr (Field == 8) {
    return getModel<void *>();
  } else if constexpr (Field == 9) {
    return getModel<Fortran::runtime::typeInfo::TypeParameterValue>();
  } else {
    llvm_unreachable("extended ISO descriptor only has 10 fields");
  }
}

} // namespace fir

#endif // OPTIMIZER_DESCRIPTOR_MODEL_H
