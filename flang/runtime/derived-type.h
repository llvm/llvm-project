//===-- runtime/derived-type.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_DERIVED_TYPE_H_
#define FORTRAN_RUNTIME_DERIVED_TYPE_H_

#include "type-code.h"
#include "flang/ISO_Fortran_binding.h"
#include <cinttypes>
#include <cstddef>

namespace Fortran::runtime {

class Descriptor;

// Static type information about derived type specializations,
// suitable for residence in read-only storage.

using TypeParameterValue = ISO::CFI_index_t;

class TypeParameter {
public:
  const char *name() const { return name_; }
  const TypeCode typeCode() const { return typeCode_; }

  bool IsLenTypeParameter() const { return which_ < 0; }

  // Returns the static value of a KIND type parameter, or the default
  // value of a LEN type parameter.
  TypeParameterValue StaticValue() const { return value_; }

  // Returns the static value of a KIND type parameter, or an
  // instantiated value of LEN type parameter.
  TypeParameterValue GetValue(const Descriptor &) const;

private:
  const char *name_;
  TypeCode typeCode_; // INTEGER, but not necessarily default kind
  int which_{-1}; // index into DescriptorAddendum LEN type parameter values
  TypeParameterValue value_; // default in the case of LEN type parameter
};

// Components that have any need for a descriptor will either reference
// a static descriptor that applies to all instances, or will *be* a
// descriptor.  Be advised: the base addresses in static descriptors
// are null.  Most runtime interfaces separate the data address from that
// of the descriptor, and ignore the encapsulated base address in the
// descriptor.  Some interfaces, e.g. calls to interoperable procedures,
// cannot pass a separate data address, and any static descriptor being used
// in that kind of situation must be copied and customized.
// Static descriptors are flagged in their attributes.
class Component {
public:
  const char *name() const { return name_; }
  TypeCode typeCode() const { return typeCode_; }
  const Descriptor *staticDescriptor() const { return staticDescriptor_; }

  bool IsParent() const { return (flags_ & PARENT) != 0; }
  bool IsPrivate() const { return (flags_ & PRIVATE) != 0; }
  bool IsDescriptor() const { return (flags_ & IS_DESCRIPTOR) != 0; }

  template <typename A> A *Locate(char *dtInstance) const {
    return reinterpret_cast<A *>(dtInstance + offset_);
  }
  template <typename A> const A *Locate(const char *dtInstance) const {
    return reinterpret_cast<const A *>(dtInstance + offset_);
  }

  Descriptor *GetDescriptor(char *dtInstance) const {
    if (IsDescriptor()) {
      return Locate<Descriptor>(dtInstance);
    } else {
      return nullptr;
    }
  }

  const Descriptor *GetDescriptor(const char *dtInstance) const {
    if (staticDescriptor_) {
      return staticDescriptor_;
    } else if (IsDescriptor()) {
      return Locate<const Descriptor>(dtInstance);
    } else {
      return nullptr;
    }
  }

private:
  enum Flag { PARENT = 1, PRIVATE = 2, IS_DESCRIPTOR = 4 };
  const char *name_{nullptr};
  std::uint32_t flags_{0};
  TypeCode typeCode_{CFI_type_other};
  const Descriptor *staticDescriptor_{nullptr};
  std::size_t offset_{0}; // byte offset in derived type instance
};

struct ExecutableCode {
  ExecutableCode() {}
  ExecutableCode(const ExecutableCode &) = default;
  ExecutableCode &operator=(const ExecutableCode &) = default;
  std::intptr_t host{0};
  std::intptr_t device{0};
};

struct TypeBoundProcedure {
  const char *name;
  ExecutableCode code;
};

// Represents a specialization of a derived type; i.e., any KIND type
// parameters have values set at compilation time.
// Extended derived types have the EXTENDS flag set and place their base
// component first in the component descriptions, which is significant for
// the execution of FINAL subroutines.
class DerivedType {
public:
  DerivedType(const char *n, std::size_t kps, std::size_t lps,
      const TypeParameter *tp, std::size_t cs, const Component *ca,
      std::size_t tbps, const TypeBoundProcedure *tbp, std::size_t sz)
      : name_{n}, kindParameters_{kps}, lenParameters_{lps}, typeParameter_{tp},
        components_{cs}, component_{ca}, typeBoundProcedures_{tbps},
        typeBoundProcedure_{tbp}, bytes_{sz} {
    if (IsNontrivialAnalysis()) {
      flags_ |= NONTRIVIAL;
    }
  }

  const char *name() const { return name_; }
  std::size_t kindParameters() const { return kindParameters_; }
  std::size_t lenParameters() const { return lenParameters_; }

  // KIND type parameters come first.
  const TypeParameter &typeParameter(int n) const { return typeParameter_[n]; }

  std::size_t components() const { return components_; }

  // The first few type-bound procedure indices are special.
  enum SpecialTBP { InitializerTBP, CopierTBP, FinalTBP };

  std::size_t typeBoundProcedures() const { return typeBoundProcedures_; }
  const TypeBoundProcedure &typeBoundProcedure(int n) const {
    return typeBoundProcedure_[n];
  }

  DerivedType &set_sequence() {
    flags_ |= SEQUENCE;
    return *this;
  }
  DerivedType &set_bind_c() {
    flags_ |= BIND_C;
    return *this;
  }

  std::size_t SizeInBytes() const { return bytes_; }
  bool Extends() const { return components_ > 0 && component_[0].IsParent(); }
  bool AnyPrivate() const;
  bool IsSequence() const { return (flags_ & SEQUENCE) != 0; }
  bool IsBindC() const { return (flags_ & BIND_C) != 0; }
  bool IsNontrivial() const { return (flags_ & NONTRIVIAL) != 0; }

  bool IsSameType(const DerivedType &) const;

  void Initialize(char *instance) const;
  void Destroy(char *instance, bool finalize = true) const;

private:
  enum Flag { SEQUENCE = 1, BIND_C = 2, NONTRIVIAL = 4 };

  // True when any descriptor of data of this derived type will require
  // an addendum pointing to a DerivedType, possibly with values of
  // LEN type parameters.  Conservative.
  bool IsNontrivialAnalysis() const;

  const char *name_{""}; // NUL-terminated constant text
  std::size_t kindParameters_{0};
  std::size_t lenParameters_{0};
  const TypeParameter *typeParameter_{nullptr}; // array
  std::size_t components_{0}; // *not* including type parameters
  const Component *component_{nullptr}; // array
  std::size_t typeBoundProcedures_{0};
  const TypeBoundProcedure *typeBoundProcedure_{nullptr}; // array
  std::uint64_t flags_{0};
  std::size_t bytes_{0};
};
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_DERIVED_TYPE_H_
