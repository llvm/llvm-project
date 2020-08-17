//===-- DescriptorModel.h -- model of descriptors for codegen ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_DESCRIPTOR_MODEL_H
#define OPTIMIZER_DESCRIPTOR_MODEL_H

#include "../runtime/descriptor.h"
#include "flang/ISO_Fortran_binding.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <tuple>

namespace fir {

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
// compiler.
//
//===----------------------------------------------------------------------===//

using TypeBuilderFunc = mlir::LLVM::LLVMType (*)(mlir::MLIRContext *);

/// Get the LLVM IR dialect model for building a particular C++ type, `T`.
template <typename T>
TypeBuilderFunc getModel();

template <>
TypeBuilderFunc getModel<void *>() {
  return [](mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMType::getInt8PtrTy(context);
  };
}
template <>
TypeBuilderFunc getModel<unsigned>() {
  return [](mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMType::getIntNTy(context, sizeof(unsigned) * 8);
  };
}
template <>
TypeBuilderFunc getModel<int>() {
  return [](mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMType::getIntNTy(context, sizeof(int) * 8);
  };
}
template <>
TypeBuilderFunc getModel<unsigned long>() {
  return [](mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMType::getIntNTy(context, sizeof(unsigned long) * 8);
  };
}
template <>
TypeBuilderFunc getModel<unsigned long long>() {
  return [](mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMType::getIntNTy(context,
                                           sizeof(unsigned long long) * 8);
  };
}
template <>
TypeBuilderFunc getModel<Fortran::ISO::CFI_rank_t>() {
  return [](mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMType::getIntNTy(
        context, sizeof(Fortran::ISO::CFI_rank_t) * 8);
  };
}
template <>
TypeBuilderFunc getModel<Fortran::ISO::CFI_type_t>() {
  return [](mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMType::getIntNTy(
        context, sizeof(Fortran::ISO::CFI_type_t) * 8);
  };
}
template <>
TypeBuilderFunc getModel<Fortran::ISO::CFI_index_t>() {
  return [](mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMType::getIntNTy(
        context, sizeof(Fortran::ISO::CFI_index_t) * 8);
  };
}
template <>
TypeBuilderFunc getModel<Fortran::ISO::CFI_dim_t>() {
  return [](mlir::MLIRContext *context) {
    auto indexTy = getModel<Fortran::ISO::CFI_index_t>()(context);
    return mlir::LLVM::LLVMType::getArrayTy(indexTy, 3);
  };
}
template <>
TypeBuilderFunc
getModel<Fortran::ISO::cfi_internal::FlexibleArray<Fortran::ISO::CFI_dim_t>>() {
  return getModel<Fortran::ISO::CFI_dim_t>();
}

//===----------------------------------------------------------------------===//
// Descriptor reflection
//===----------------------------------------------------------------------===//

/// Get the type model of the field number `Field` in an ISO descriptor.
template <int Field>
static constexpr TypeBuilderFunc getDescFieldTypeModel() {
  Fortran::ISO::Fortran_2018::CFI_cdesc_t dummyDesc{};
  // check that the descriptor is exactly 8 fields
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
    return getModel<std::uint64_t>();
  } else if constexpr (Field == 10) {
    return getModel<Fortran::runtime::TypeParameterValue>();
  } else {
    llvm_unreachable("extended ISO descriptor only has 11 fields");
  }
}

} // namespace fir

#endif // OPTIMIZER_DESCRIPTOR_MODEL_H
