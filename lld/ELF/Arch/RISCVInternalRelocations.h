//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_ARCH_RISCVINTERNALRELOCATIONS_H
#define LLD_ELF_ARCH_RISCVINTERNALRELOCATIONS_H

#include "Relocations.h"
#include "Symbols.h"

namespace lld::elf {

// Bit 8 of RelType is used to indicate linker-internal relocations that are
// not vendor-specific.
// These are internal relocation numbers for GP/X0 relaxation. They aren't part
// of the psABI spec.
constexpr uint32_t INTERNAL_R_RISCV_GPREL_I = 256;
constexpr uint32_t INTERNAL_R_RISCV_GPREL_S = 257;
constexpr uint32_t INTERNAL_R_RISCV_X0REL_I = 258;
constexpr uint32_t INTERNAL_R_RISCV_X0REL_S = 259;

// Bits 9 -> 31 of RelType are used to indicate vendor-specific relocations.
constexpr uint32_t INTERNAL_RISCV_VENDOR_MASK = 0xFFFFFFFF << 9;
constexpr uint32_t INTERNAL_RISCV_VENDOR_QUALCOMM = 1 << 9;
constexpr uint32_t INTERNAL_RISCV_VENDOR_ANDES = 2 << 9;

constexpr uint32_t INTERNAL_RISCV_QC_ABS20_U =
    INTERNAL_RISCV_VENDOR_QUALCOMM | llvm::ELF::R_RISCV_QC_ABS20_U;
constexpr uint32_t INTERNAL_RISCV_QC_E_BRANCH =
    INTERNAL_RISCV_VENDOR_QUALCOMM | llvm::ELF::R_RISCV_QC_E_BRANCH;
constexpr uint32_t INTERNAL_RISCV_QC_E_32 =
    INTERNAL_RISCV_VENDOR_QUALCOMM | llvm::ELF::R_RISCV_QC_E_32;
constexpr uint32_t INTERNAL_RISCV_QC_E_CALL_PLT =
    INTERNAL_RISCV_VENDOR_QUALCOMM | llvm::ELF::R_RISCV_QC_E_CALL_PLT;

constexpr uint32_t INTERNAL_RISCV_NDS_BRANCH_10 =
    INTERNAL_RISCV_VENDOR_ANDES | llvm::ELF::R_RISCV_NDS_BRANCH_10;

uint32_t getRISCVVendorRelMarker(llvm::StringRef rvVendor);
std::optional<llvm::StringRef> getRISCVVendorString(RelType ty);

class vendor_reloc_iterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = Relocation;
  using difference_type = std::ptrdiff_t;
  using pointer = Relocation *;
  using reference = Relocation; // returned by value

  vendor_reloc_iterator(MutableArrayRef<Relocation>::iterator i,
                        MutableArrayRef<Relocation>::iterator e)
      : it(i), end(e) {}

  // Dereference
  Relocation operator*() const {
    Relocation r = *it;
    r.type.v |= rvVendorFlag;
    return r;
  }

  struct vendor_reloc_proxy {
    Relocation r;
    const Relocation *operator->() const { return &r; }
  };

  vendor_reloc_proxy operator->() const {
    return vendor_reloc_proxy{this->operator*()};
  }

  vendor_reloc_iterator &operator++() {
    ++it;
    if (it != end && it->type == llvm::ELF::R_RISCV_VENDOR) {
      rvVendorFlag = getRISCVVendorRelMarker(it->sym->getName());
      ++it;
    } else {
      rvVendorFlag = 0;
    }
    return *this;
  }

  vendor_reloc_iterator operator++(int) {
    vendor_reloc_iterator tmp(*this);
    ++(*this);
    return tmp;
  }

  bool operator==(const vendor_reloc_iterator &other) const {
    return it == other.it;
  }
  bool operator!=(const vendor_reloc_iterator &other) const {
    return it != other.it;
  }

  Relocation *getUnderlyingRelocation() const { return &*it; }

private:
  MutableArrayRef<Relocation>::iterator it;
  MutableArrayRef<Relocation>::iterator end;
  uint32_t rvVendorFlag = 0;
};

inline auto riscv_vendor_relocs(MutableArrayRef<Relocation> arr) {
  return llvm::make_range(vendor_reloc_iterator(arr.begin(), arr.end()),
                          vendor_reloc_iterator(arr.end(), arr.end()));
}

} // namespace lld::elf

#endif
