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

namespace Fortran::runtime::typeInfo {

class DerivedType {
public:
  ~DerivedType();

  // This member comes first because it's used like a vtable by generated code.
  // It includes all of the ancestor types' bindings, if any, first,
  // with any overrides from descendants already applied to them.  Local
  // bindings then follow in alphabetic order of binding name.
  StaticDescriptor<1, true>
      binding; // TYPE(BINDING), DIMENSION(:), POINTER, CONTIGUOUS

  StaticDescriptor<0> name; // CHARACTER(:), POINTER

  std::uint64_t sizeInBytes{0};
  StaticDescriptor<0, true> parent; // TYPE(DERIVEDTYPE), POINTER

  // Instantiations of a parameterized derived type with KIND type
  // parameters will point this data member to the description of
  // the original uninstantiated type, which may be shared from a
  // module via use association.  The original uninstantiated derived
  // type description will point to itself.  Derived types that have
  // no KIND type parameters will have a null pointer here.
  StaticDescriptor<0, true> uninstantiated; // TYPE(DERIVEDTYPE), POINTER

  // TODO: flags for SEQUENCE, BIND(C), any PRIVATE component(? see 7.5.2)
  std::uint64_t typeHash{0};

  // These pointer targets include all of the items from the parent, if any.
  StaticDescriptor<1> kindParameter; // pointer to rank-1 array of INTEGER(8)
  StaticDescriptor<1> lenParameterKind; // pointer to rank-1 array of INTEGER(1)

  // This array of local data components includes the parent component.
  // Components are in alphabetic order.
  // It does not include procedure pointer components.
  StaticDescriptor<1, true>
      component; // TYPE(COMPONENT), POINTER, DIMENSION(:), CONTIGUOUS

  // Procedure pointer components
  StaticDescriptor<1, true>
      procPtr; // TYPE(PROCPTR), POINTER, DIMENSION(:), CONTIGUOUS

  // Does not include special bindings from ancestral types.
  StaticDescriptor<1, true>
      special; // TYPE(SPECIALBINDING), POINTER, DIMENSION(:), CONTIGUOUS

  std::size_t LenParameters() const {
    return lenParameterKind.descriptor().Elements();
  }
};

using ProcedurePointer = void (*)(); // TYPE(C_FUNPTR)

struct Binding {
  ProcedurePointer proc;
  StaticDescriptor<0> name; // CHARACTER(:), POINTER
};

struct Value {
  enum class Genre : std::uint8_t {
    Deferred = 1,
    Explicit = 2,
    LenParameter = 3
  };
  Genre genre{Genre::Explicit};
  // The value encodes an index into the table of LEN type parameters in
  // a descriptor's addendum for genre == Genre::LenParameter.
  TypeParameterValue value{0};
};

struct Component {
  enum class Genre : std::uint8_t { Data, Pointer, Allocatable, Automatic };
  StaticDescriptor<0> name; // CHARACTER(:), POINTER
  Genre genre{Genre::Data};
  std::uint8_t category; // common::TypeCategory
  std::uint8_t kind{0};
  std::uint8_t rank{0};
  std::uint64_t offset{0};
  Value characterLen; // for TypeCategory::Character
  StaticDescriptor<0, true> derivedType; // TYPE(DERIVEDTYPE), POINTER
  StaticDescriptor<1, true>
      lenValue; // TYPE(VALUE), POINTER, DIMENSION(:), CONTIGUOUS
  StaticDescriptor<2, true>
      bounds; // TYPE(VALUE), POINTER, DIMENSION(2,:), CONTIGUOUS
  char *initialization{nullptr}; // for Genre::Data and Pointer
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
