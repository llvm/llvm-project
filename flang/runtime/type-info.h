//===-- runtime/type-info.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_TYPE_INFO_H_
#define FORTRAN_RUNTIME_TYPE_INFO_H_

// A C++ perspective of the derived type description schemata in
// flang/module/__fortran_type_info.f90.

#include "descriptor.h"
#include "flang/Common/Fortran.h"
#include <cinttypes>
#include <memory>
#include <optional>

namespace Fortran::runtime::typeInfo {

class Component;

class DerivedType {
public:
  ~DerivedType(); // never defined

  const Descriptor &binding() const { return binding_.descriptor(); }
  const Descriptor &name() const { return name_.descriptor(); }
  std::uint64_t sizeInBytes() const { return sizeInBytes_; }
  const Descriptor &parent() const { return parent_.descriptor(); }
  std::uint64_t typeHash() const { return typeHash_; }
  const Descriptor &uninstatiated() const {
    return uninstantiated_.descriptor();
  }
  const Descriptor &kindParameter() const {
    return kindParameter_.descriptor();
  }
  const Descriptor &lenParameterKind() const {
    return lenParameterKind_.descriptor();
  }
  const Descriptor &component() const { return component_.descriptor(); }
  const Descriptor &procPtr() const { return procPtr_.descriptor(); }
  const Descriptor &special() const { return special_.descriptor(); }

  std::size_t LenParameters() const { return lenParameterKind().Elements(); }

  // Finds a data component by name in this derived type or tis ancestors.
  const Component *FindDataComponent(
      const char *name, std::size_t nameLen) const;

  FILE *Dump(FILE * = stdout) const;

private:
  // This member comes first because it's used like a vtable by generated code.
  // It includes all of the ancestor types' bindings, if any, first,
  // with any overrides from descendants already applied to them.  Local
  // bindings then follow in alphabetic order of binding name.
  StaticDescriptor<1, true>
      binding_; // TYPE(BINDING), DIMENSION(:), POINTER, CONTIGUOUS

  StaticDescriptor<0> name_; // CHARACTER(:), POINTER

  std::uint64_t sizeInBytes_{0};
  StaticDescriptor<0, true> parent_; // TYPE(DERIVEDTYPE), POINTER

  // Instantiations of a parameterized derived type with KIND type
  // parameters will point this data member to the description of
  // the original uninstantiated type, which may be shared from a
  // module via use association.  The original uninstantiated derived
  // type description will point to itself.  Derived types that have
  // no KIND type parameters will have a null pointer here.
  StaticDescriptor<0, true> uninstantiated_; // TYPE(DERIVEDTYPE), POINTER

  // TODO: flags for SEQUENCE, BIND(C), any PRIVATE component(? see 7.5.2)
  std::uint64_t typeHash_{0};

  // These pointer targets include all of the items from the parent, if any.
  StaticDescriptor<1> kindParameter_; // pointer to rank-1 array of INTEGER(8)
  StaticDescriptor<1>
      lenParameterKind_; // pointer to rank-1 array of INTEGER(1)

  // This array of local data components includes the parent component.
  // Components are in component order, not collation order of their names.
  // It does not include procedure pointer components.
  StaticDescriptor<1, true>
      component_; // TYPE(COMPONENT), POINTER, DIMENSION(:), CONTIGUOUS

  // Procedure pointer components
  StaticDescriptor<1, true>
      procPtr_; // TYPE(PROCPTR), POINTER, DIMENSION(:), CONTIGUOUS

  // Does not include special bindings from ancestral types.
  StaticDescriptor<1, true>
      special_; // TYPE(SPECIALBINDING), POINTER, DIMENSION(:), CONTIGUOUS
};

using ProcedurePointer = void (*)(); // TYPE(C_FUNPTR)

struct Binding {
  ProcedurePointer proc;
  StaticDescriptor<0> name; // CHARACTER(:), POINTER
};

class Value {
public:
  enum class Genre : std::uint8_t {
    Deferred = 1,
    Explicit = 2,
    LenParameter = 3
  };

  std::optional<TypeParameterValue> GetValue(const Descriptor *) const;

private:
  Genre genre_{Genre::Explicit};
  // The value encodes an index into the table of LEN type parameters in
  // a descriptor's addendum for genre == Genre::LenParameter.
  TypeParameterValue value_{0};
};

class Component {
public:
  enum class Genre : std::uint8_t {
    Data = 1,
    Pointer = 2,
    Allocatable = 3,
    Automatic = 4
  };

  const Descriptor &name() const { return name_.descriptor(); }
  Genre genre() const { return genre_; }
  TypeCategory category() const { return static_cast<TypeCategory>(category_); }
  int kind() const { return kind_; }
  int rank() const { return rank_; }
  std::uint64_t offset() const { return offset_; }
  const Value &characterLen() const { return characterLen_; }
  const DerivedType *derivedType() const {
    return derivedType_.descriptor().OffsetElement<const DerivedType>();
  }
  const Value *lenValue() const {
    return lenValue_.descriptor().OffsetElement<const Value>();
  }
  const Value *bounds() const {
    return bounds_.descriptor().OffsetElement<const Value>();
  }
  const char *initialization() const { return initialization_; }

  // Creates a pointer descriptor from a component description.
  void EstablishDescriptor(Descriptor &, const Descriptor &container,
      const SubscriptValue[], Terminator &) const;

  FILE *Dump(FILE * = stdout) const;

private:
  StaticDescriptor<0> name_; // CHARACTER(:), POINTER
  Genre genre_{Genre::Data};
  std::uint8_t category_; // common::TypeCategory
  std::uint8_t kind_{0};
  std::uint8_t rank_{0};
  std::uint64_t offset_{0};
  Value characterLen_; // for TypeCategory::Character
  StaticDescriptor<0, true> derivedType_; // TYPE(DERIVEDTYPE), POINTER
  StaticDescriptor<1, true>
      lenValue_; // TYPE(VALUE), POINTER, DIMENSION(:), CONTIGUOUS
  StaticDescriptor<2, true>
      bounds_; // TYPE(VALUE), POINTER, DIMENSION(2,:), CONTIGUOUS
  const char *initialization_{nullptr}; // for Genre::Data and Pointer
  // TODO: cobounds
  // TODO: `PRIVATE` attribute
};

struct ProcPtrComponent {
  StaticDescriptor<0> name; // CHARACTER(:), POINTER
  std::uint64_t offset{0};
  ProcedurePointer procInitialization; // for Genre::Procedure
};

struct SpecialBinding {
  enum class Which : std::uint8_t {
    None = 0,
    Assignment = 4,
    ElementalAssignment = 5,
    Final = 8,
    ElementalFinal = 9,
    AssumedRankFinal = 10,
    ReadFormatted = 16,
    ReadUnformatted = 17,
    WriteFormatted = 18,
    WriteUnformatted = 19
  } which{Which::None};

  // Used for Which::Final only.  Which::Assignment always has rank 0, as
  // type-bound defined assignment for rank > 0 must be elemental
  // due to the required passed object dummy argument, which are scalar.
  // User defined derived type I/O is always scalar.
  std::uint8_t rank{0};

  // The following little bit-set identifies which dummy arguments are
  // passed via descriptors for their derived type arguments.
  //   Which::Assignment and Which::ElementalAssignment:
  //     Set to 1, 2, or (usually 3).
  //     The passed-object argument (usually the "to") is always passed via a
  //     a descriptor in the cases where the runtime will call a defined
  //     assignment because these calls are to type-bound generics,
  //     not generic interfaces, and type-bound generic defined assigment
  //     may appear only in an extensible type and requires a passed-object
  //     argument (see C774), and passed-object arguments to TBPs must be
  //     both polymorphic and scalar (C760).  The non-passed-object argument
  //     (usually the "from") is usually, but not always, also a descriptor.
  //   Which::Final and Which::ElementalFinal:
  //     Set to 1 when dummy argument is assumed-shape; otherwise, the
  //     argument can be passed by address.  (Fortran guarantees that
  //     any finalized object must be whole and contiguous by restricting
  //     the use of DEALLOCATE on pointers.  The dummy argument of an
  //     elemental final subroutine must be scalar and monomorphic, but
  //     use a descriptors when the type has LEN parameters.)
  //   Which::AssumedRankFinal: flag must necessarily be set
  //   User derived type I/O:
  //     Set to 1 when "dtv" initial dummy argument is polymorphic, which is
  //     the case when and only when the derived type is extensible.
  //     When false, the user derived type I/O subroutine must have been
  //     called via a generic interface, not a generic TBP.
  std::uint8_t isArgDescriptorSet{0};

  ProcedurePointer proc{nullptr};
};
} // namespace Fortran::runtime::typeInfo
#endif // FORTRAN_RUNTIME_TYPE_INFO_H_
