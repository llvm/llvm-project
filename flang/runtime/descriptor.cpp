//===-- runtime/descriptor.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "descriptor.h"
#include "derived.h"
#include "memory.h"
#include "stat.h"
#include "terminator.h"
#include "type-info.h"
#include <cassert>
#include <cstdlib>
#include <cstring>

namespace Fortran::runtime {

Descriptor::Descriptor(const Descriptor &that) { *this = that; }

Descriptor &Descriptor::operator=(const Descriptor &that) {
  std::memcpy(this, &that, that.SizeInBytes());
  return *this;
}

void Descriptor::Establish(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute,
    bool addendum) {
  Terminator terminator{__FILE__, __LINE__};
  // Subtle: the standard CFI_establish() function doesn't allow a zero
  // elem_len argument in cases where elem_len is not ignored; and when it
  // returns an error code (CFI_INVALID_ELEM_LEN in this case), it must not
  // modify the descriptor.  That design makes sense, maybe, for actual
  // C interoperability, but we need to work around it here.  A zero
  // incoming element length is replaced by 4 so that it will be valid
  // for all CHARACTER kinds.
  std::size_t workaroundElemLen{elementBytes ? elementBytes : 4};
  int cfiStatus{ISO::CFI_establish(
      &raw_, p, attribute, t.raw(), workaroundElemLen, rank, extent)};
  if (cfiStatus != CFI_SUCCESS) {
    terminator.Crash(
        "Descriptor::Establish: CFI_establish returned %d", cfiStatus, t.raw());
  }
  if (elementBytes == 0) {
    raw_.elem_len = 0;
    for (int j{0}; j < rank; ++j) {
      GetDimension(j).SetByteStride(0);
    }
  }
  raw_.f18Addendum = addendum;
  DescriptorAddendum *a{Addendum()};
  RUNTIME_CHECK(terminator, addendum == (a != nullptr));
  if (a) {
    new (a) DescriptorAddendum{};
  }
}

void Descriptor::Establish(TypeCategory c, int kind, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute,
    bool addendum) {
  Establish(TypeCode(c, kind), BytesFor(c, kind), p, rank, extent, attribute,
      addendum);
}

void Descriptor::Establish(int characterKind, std::size_t characters, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute,
    bool addendum) {
  Establish(TypeCode{TypeCategory::Character, characterKind},
      characterKind * characters, p, rank, extent, attribute, addendum);
}

void Descriptor::Establish(const typeInfo::DerivedType &dt, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  Establish(TypeCode{TypeCategory::Derived, 0}, dt.sizeInBytes(), p, rank,
      extent, attribute, true);
  DescriptorAddendum *a{Addendum()};
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator, a != nullptr);
  new (a) DescriptorAddendum{&dt};
}

OwningPtr<Descriptor> Descriptor::Create(TypeCode t, std::size_t elementBytes,
    void *p, int rank, const SubscriptValue *extent,
    ISO::CFI_attribute_t attribute, int derivedTypeLenParameters) {
  std::size_t bytes{SizeInBytes(rank, true, derivedTypeLenParameters)};
  Terminator terminator{__FILE__, __LINE__};
  Descriptor *result{
      reinterpret_cast<Descriptor *>(AllocateMemoryOrCrash(terminator, bytes))};
  result->Establish(t, elementBytes, p, rank, extent, attribute, true);
  return OwningPtr<Descriptor>{result};
}

OwningPtr<Descriptor> Descriptor::Create(TypeCategory c, int kind, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  return Create(
      TypeCode(c, kind), BytesFor(c, kind), p, rank, extent, attribute);
}

OwningPtr<Descriptor> Descriptor::Create(int characterKind,
    SubscriptValue characters, void *p, int rank, const SubscriptValue *extent,
    ISO::CFI_attribute_t attribute) {
  return Create(TypeCode{TypeCategory::Character, characterKind},
      characterKind * characters, p, rank, extent, attribute);
}

OwningPtr<Descriptor> Descriptor::Create(const typeInfo::DerivedType &dt,
    void *p, int rank, const SubscriptValue *extent,
    ISO::CFI_attribute_t attribute) {
  return Create(TypeCode{TypeCategory::Derived, 0}, dt.sizeInBytes(), p, rank,
      extent, attribute, dt.LenParameters());
}

std::size_t Descriptor::SizeInBytes() const {
  const DescriptorAddendum *addendum{Addendum()};
  return sizeof *this - sizeof(Dimension) + raw_.rank * sizeof(Dimension) +
      (addendum ? addendum->SizeInBytes() : 0);
}

std::size_t Descriptor::Elements() const {
  int n{rank()};
  std::size_t elements{1};
  for (int j{0}; j < n; ++j) {
    elements *= GetDimension(j).Extent();
  }
  return elements;
}

int Descriptor::Allocate() {
  std::size_t byteSize{Elements() * ElementBytes()};
  void *p{std::malloc(byteSize)};
  if (!p && byteSize) {
    return CFI_ERROR_MEM_ALLOCATION;
  }
  // TODO: image synchronization
  raw_.base_addr = p;
  if (int dims{rank()}) {
    std::size_t stride{ElementBytes()};
    for (int j{0}; j < dims; ++j) {
      auto &dimension{GetDimension(j)};
      dimension.SetByteStride(stride);
      stride *= dimension.Extent();
    }
  }
  return 0;
}

int Descriptor::Destroy(bool finalize) {
  if (raw_.attribute == CFI_attribute_pointer) {
    return StatOk;
  } else {
    if (auto *addendum{Addendum()}) {
      if (const auto *derived{addendum->derivedType()}) {
        if (!derived->noDestructionNeeded()) {
          runtime::Destroy(*this, finalize, *derived);
        }
      }
    }
    return Deallocate();
  }
}

int Descriptor::Deallocate() { return ISO::CFI_deallocate(&raw_); }

bool Descriptor::IncrementSubscripts(
    SubscriptValue *subscript, const int *permutation) const {
  for (int j{0}; j < raw_.rank; ++j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    if (subscript[k]++ < dim.UpperBound()) {
      return true;
    }
    subscript[k] = dim.LowerBound();
  }
  return false;
}

bool Descriptor::DecrementSubscripts(
    SubscriptValue *subscript, const int *permutation) const {
  for (int j{raw_.rank - 1}; j >= 0; --j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    if (--subscript[k] >= dim.LowerBound()) {
      return true;
    }
    subscript[k] = dim.UpperBound();
  }
  return false;
}

std::size_t Descriptor::ZeroBasedElementNumber(
    const SubscriptValue *subscript, const int *permutation) const {
  std::size_t result{0};
  std::size_t coefficient{1};
  for (int j{0}; j < raw_.rank; ++j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    result += coefficient * (subscript[k] - dim.LowerBound());
    coefficient *= dim.Extent();
  }
  return result;
}

bool Descriptor::SubscriptsForZeroBasedElementNumber(SubscriptValue *subscript,
    std::size_t elementNumber, const int *permutation) const {
  std::size_t coefficient{1};
  std::size_t dimCoefficient[maxRank];
  for (int j{0}; j < raw_.rank; ++j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    dimCoefficient[j] = coefficient;
    coefficient *= dim.Extent();
  }
  if (elementNumber >= coefficient) {
    return false; // out of range
  }
  for (int j{raw_.rank - 1}; j >= 0; --j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    std::size_t quotient{elementNumber / dimCoefficient[j]};
    subscript[k] = quotient + dim.LowerBound();
    elementNumber -= quotient * dimCoefficient[j];
  }
  return true;
}

bool Descriptor::EstablishPointerSection(const Descriptor &source,
    const SubscriptValue *lower, const SubscriptValue *upper,
    const SubscriptValue *stride) {
  *this = source;
  raw_.attribute = CFI_attribute_pointer;
  int newRank{raw_.rank};
  for (int j{0}; j < raw_.rank; ++j) {
    if (!stride || stride[j] == 0) {
      if (newRank > 0) {
        --newRank;
      } else {
        return false;
      }
    }
  }
  raw_.rank = newRank;
  return CFI_section(&raw_, &source.raw_, lower, upper, stride) == CFI_SUCCESS;
}

void Descriptor::Check() const {
  // TODO
}

void Descriptor::Dump(FILE *f) const {
  std::fprintf(f, "Descriptor @ %p:\n", reinterpret_cast<const void *>(this));
  std::fprintf(f, "  base_addr %p\n", raw_.base_addr);
  std::fprintf(f, "  elem_len  %zd\n", static_cast<std::size_t>(raw_.elem_len));
  std::fprintf(f, "  version   %d\n", static_cast<int>(raw_.version));
  std::fprintf(f, "  rank      %d\n", static_cast<int>(raw_.rank));
  std::fprintf(f, "  type      %d\n", static_cast<int>(raw_.type));
  std::fprintf(f, "  attribute %d\n", static_cast<int>(raw_.attribute));
  std::fprintf(f, "  addendum  %d\n", static_cast<int>(raw_.f18Addendum));
  for (int j{0}; j < raw_.rank; ++j) {
    std::fprintf(f, "  dim[%d] lower_bound %jd\n", j,
        static_cast<std::intmax_t>(raw_.dim[j].lower_bound));
    std::fprintf(f, "         extent      %jd\n",
        static_cast<std::intmax_t>(raw_.dim[j].extent));
    std::fprintf(f, "         sm          %jd\n",
        static_cast<std::intmax_t>(raw_.dim[j].sm));
  }
  if (const DescriptorAddendum * addendum{Addendum()}) {
    addendum->Dump(f);
  }
}

DescriptorAddendum &DescriptorAddendum::operator=(
    const DescriptorAddendum &that) {
  derivedType_ = that.derivedType_;
  auto lenParms{that.LenParameters()};
  for (std::size_t j{0}; j < lenParms; ++j) {
    len_[j] = that.len_[j];
  }
  return *this;
}

std::size_t DescriptorAddendum::SizeInBytes() const {
  return SizeInBytes(LenParameters());
}

std::size_t DescriptorAddendum::LenParameters() const {
  const auto *type{derivedType()};
  return type ? type->LenParameters() : 0;
}

void DescriptorAddendum::Dump(FILE *f) const {
  std::fprintf(
      f, "  derivedType @ %p\n", reinterpret_cast<const void *>(derivedType()));
  std::size_t lenParms{LenParameters()};
  for (std::size_t j{0}; j < lenParms; ++j) {
    std::fprintf(f, "  len[%zd] %jd\n", j, static_cast<std::intmax_t>(len_[j]));
  }
}
} // namespace Fortran::runtime
