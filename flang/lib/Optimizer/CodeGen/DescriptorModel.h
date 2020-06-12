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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/ErrorHandling.h"
#include <tuple>

namespace fir {

//===----------------------------------------------------------------------===//
// Static size information
//===----------------------------------------------------------------------===//

static constexpr std::size_t sizeOfDimElement() {
  return sizeof(Fortran::ISO::Fortran_2018::CFI_index_t);
}
static constexpr std::size_t sizeOfDimRow() {
  return sizeof(Fortran::ISO::Fortran_2018::CFI_dim_t);
}
static constexpr std::size_t sizeOfBareDesc() {
  return sizeof(Fortran::ISO::Fortran_2018::CFI_cdesc_t);
}
static constexpr std::size_t sizeOfDesc(unsigned rank) {
  return sizeOfBareDesc() + rank * sizeOfDimRow();
}
static constexpr std::size_t sizeOfTypeParam() {
  return sizeof(Fortran::runtime::TypeParameterValue);
}
static constexpr std::size_t sizeOfDescAddendum() {
  return sizeof(Fortran::runtime::DescriptorAddendum);
}
static constexpr std::size_t sizeOfExtendedDesc(unsigned rank,
                                                unsigned lenParams) {
  return sizeOfDesc(rank) + sizeOfDescAddendum() +
         lenParams * sizeOfTypeParam();
}

//===----------------------------------------------------------------------===//
// Descriptor reflection
//
// This supplies a set of model builders to decompose the C declaration of a
// descriptor (as encoded in ISO_Fortran_binding.h and elsewhere) and
// reconstruct that type in the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

using TypeBuilderFunc = mlir::LLVM::LLVMType (*)(mlir::LLVM::LLVMDialect *);

template <typename T>
TypeBuilderFunc getModel();
template <>
TypeBuilderFunc getModel<void *>() {
  return [](mlir::LLVM::LLVMDialect *dialect) {
    return mlir::LLVM::LLVMType::getInt8PtrTy(dialect);
  };
}
template <>
TypeBuilderFunc getModel<uint32_t>() {
  return [](mlir::LLVM::LLVMDialect *dialect) {
    return mlir::LLVM::LLVMType::getIntNTy(dialect, sizeof(uint32_t) * 8);
  };
}
template <>
TypeBuilderFunc getModel<int>() {
  return [](mlir::LLVM::LLVMDialect *dialect) {
    return mlir::LLVM::LLVMType::getIntNTy(dialect, sizeof(int) * 8);
  };
}
template <>
TypeBuilderFunc getModel<uint64_t>() {
  return [](mlir::LLVM::LLVMDialect *dialect) {
    return mlir::LLVM::LLVMType::getIntNTy(dialect, sizeof(uint64_t) * 8);
  };
}
template <>
TypeBuilderFunc getModel<Fortran::ISO::CFI_rank_t>() {
  return [](mlir::LLVM::LLVMDialect *dialect) {
    return mlir::LLVM::LLVMType::getIntNTy(
        dialect, sizeof(Fortran::ISO::CFI_rank_t) * 8);
  };
}
template <>
TypeBuilderFunc getModel<Fortran::ISO::CFI_type_t>() {
  return [](mlir::LLVM::LLVMDialect *dialect) {
    return mlir::LLVM::LLVMType::getIntNTy(
        dialect, sizeof(Fortran::ISO::CFI_type_t) * 8);
  };
}
template <>
TypeBuilderFunc getModel<Fortran::ISO::CFI_index_t>() {
  return [](mlir::LLVM::LLVMDialect *dialect) {
    return mlir::LLVM::LLVMType::getIntNTy(
        dialect, sizeof(Fortran::ISO::CFI_index_t) * 8);
  };
}
template <>
TypeBuilderFunc getModel<Fortran::ISO::CFI_dim_t>() {
  return [](mlir::LLVM::LLVMDialect *dialect) {
    auto indexTy = getModel<Fortran::ISO::CFI_index_t>()(dialect);
    return mlir::LLVM::LLVMType::getArrayTy(indexTy, 3);
  };
}
template <>
TypeBuilderFunc
getModel<Fortran::ISO::cfi_internal::FlexibleArray<Fortran::ISO::CFI_dim_t>>() {
  return getModel<Fortran::ISO::CFI_dim_t>();
}

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
