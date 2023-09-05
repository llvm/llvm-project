//===-- include/flang/Common/Fortran.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_FORTRAN_H_
#define FORTRAN_COMMON_FORTRAN_H_

// Fortran language concepts that are used in many phases are defined
// once here to avoid redundancy and needless translation.

#include "enum-set.h"
#include "idioms.h"
#include <cinttypes>
#include <optional>
#include <string>

namespace Fortran::common {

// Fortran has five kinds of intrinsic data types, plus the derived types.
ENUM_CLASS(TypeCategory, Integer, Real, Complex, Character, Logical, Derived)
ENUM_CLASS(VectorElementCategory, Integer, Unsigned, Real)

constexpr bool IsNumericTypeCategory(TypeCategory category) {
  return category == TypeCategory::Integer || category == TypeCategory::Real ||
      category == TypeCategory::Complex;
}

// Kinds of IMPORT statements. Default means IMPORT or IMPORT :: names.
ENUM_CLASS(ImportKind, Default, Only, None, All)

// The attribute on a type parameter can be KIND or LEN.
ENUM_CLASS(TypeParamAttr, Kind, Len)

ENUM_CLASS(NumericOperator, Power, Multiply, Divide, Add, Subtract)
const char *AsFortran(NumericOperator);

ENUM_CLASS(LogicalOperator, And, Or, Eqv, Neqv, Not)
const char *AsFortran(LogicalOperator);

ENUM_CLASS(RelationalOperator, LT, LE, EQ, NE, GE, GT)
const char *AsFortran(RelationalOperator);

ENUM_CLASS(Intent, Default, In, Out, InOut)

ENUM_CLASS(IoStmtKind, None, Backspace, Close, Endfile, Flush, Inquire, Open,
    Print, Read, Rewind, Wait, Write)

// Union of specifiers for all I/O statements.
ENUM_CLASS(IoSpecKind, Access, Action, Advance, Asynchronous, Blank, Decimal,
    Delim, Direct, Encoding, End, Eor, Err, Exist, File, Fmt, Form, Formatted,
    Id, Iomsg, Iostat, Name, Named, Newunit, Nextrec, Nml, Number, Opened, Pad,
    Pending, Pos, Position, Read, Readwrite, Rec, Recl, Round, Sequential, Sign,
    Size, Status, Stream, Unformatted, Unit, Write,
    Carriagecontrol, // nonstandard
    Convert, // nonstandard
    Dispose, // nonstandard
)

// Defined I/O variants
ENUM_CLASS(
    DefinedIo, ReadFormatted, ReadUnformatted, WriteFormatted, WriteUnformatted)
const char *AsFortran(DefinedIo);

// Floating-point rounding modes; these are packed into a byte to save
// room in the runtime's format processing context structure.
enum class RoundingMode : std::uint8_t {
  TiesToEven, // ROUND=NEAREST, RN - default IEEE rounding
  ToZero, // ROUND=ZERO, RZ - truncation
  Down, // ROUND=DOWN, RD
  Up, // ROUND=UP, RU
  TiesAwayFromZero, // ROUND=COMPATIBLE, RC - ties round away from zero
};

ENUM_CLASS(IntrinsicOperator, Power, Multiply, Divide, Add, Subtract, Concat,
    LT, LE, EQ, NE, GE, GT, NOT, AND, OR, EQV, NEQV)

ENUM_CLASS(AccessSpecKind, Public, Private)

ENUM_CLASS(IntentSpecKind, In, Out, InOut)

ENUM_CLASS(BindEntityKind, Object, Common)

ENUM_CLASS(SavedEntityKind, Entity, Common)

ENUM_CLASS(ImplicitNoneNameSpec, External, Type) // R866

ENUM_CLASS(StopKind, Stop, ErrorStop)

ENUM_CLASS(ConnectCharExprKind, Access, Action, Asynchronous, Blank, Decimal,
    Delim, Encoding, Form, Pad, Position, Round, Sign,
    /* extensions: */ Carriagecontrol, Convert, Dispose)

ENUM_CLASS(
    IoControlCharExprKind, Advance, Blank, Decimal, Delim, Pad, Round, Sign)

ENUM_CLASS(InquireCharVarKind, Access, Action, Asynchronous, Blank, Decimal,
    Delim, Direct, Encoding, Form, Formatted, Iomsg, Name, Pad, Position, Read,
    Readwrite, Round, Sequential, Sign, Stream, Status, Unformatted, Write,
    /* extensions: */ Carriagecontrol, Convert, Dispose)

ENUM_CLASS(InquireIntVarKind, Iostat, Nextrec, Number, Pos, Recl, Size)

ENUM_CLASS(InquireLogVarKind, Exist, Named, Opened, Pending)

ENUM_CLASS(ModuleNature, Intrinsic, Non_Intrinsic) // R1410

ENUM_CLASS(ProcedureKind, ModuleProcedure, Procedure)

// OpenMP kinds
ENUM_CLASS(OmpProcBindClauseKind, Close, Master, Spread, Primary)

ENUM_CLASS(OmpDefaultClauseKind, Private, Firstprivate, Shared, None)

ENUM_CLASS(OmpMapKind, To, From, Tofrom, Alloc, Release, Delete)

ENUM_CLASS(OmpDefaultmapClauseImplicitBehavior, Alloc, To, From, Tofrom,
    Firstprivate, None, Default)

ENUM_CLASS(OmpDefaultmapClauseVariableCategory, Scalar, Aggregate, Allocatable,
    Pointer)

ENUM_CLASS(OmpScheduleModifierKind, Monotonic, Nonmonotonic, Simd)

ENUM_CLASS(OmpScheduleClauseKind, Static, Dynamic, Guided, Auto, Runtime)

ENUM_CLASS(OmpDeviceClauseDeviceModifier, Ancestor, Device_Num)

ENUM_CLASS(OmpDeviceTypeClauseKind, Any, Host, Nohost)

ENUM_CLASS(OmpIfClauseDirectiveNameModifier, Parallel, Simd, Target, TargetData,
    TargetEnterData, TargetExitData, TargetUpdate, Task, Taskloop, Teams)

ENUM_CLASS(OmpOrderModifierKind, Reproducible, Unconstrained)

ENUM_CLASS(OmpOrderClauseKind, Concurrent)

ENUM_CLASS(OmpLinearModifierKind, Ref, Val, Uval)

ENUM_CLASS(OmpDependenceKind, In, Out, Inout, Source, Sink)

ENUM_CLASS(OmpAtomicDefaultMemOrderClauseKind, SeqCst, AcqRel, Relaxed)

ENUM_CLASS(OmpCancelKind, Parallel, Sections, Do, Taskgroup)

// OpenACC kinds
ENUM_CLASS(AccDataModifierKind, ReadOnly, Zero)

ENUM_CLASS(AccReductionOperatorKind, Plus, Multiply, Max, Min, Iand, Ior, Ieor,
    And, Or, Eqv, Neqv)

// Fortran label. Must be in [1..99999].
using Label = std::uint64_t;

// Fortran arrays may have up to 15 dimensions (See Fortran 2018 section 5.4.6).
static constexpr int maxRank{15};

// CUDA subprogram attribute combinations
ENUM_CLASS(CUDASubprogramAttrs, Host, Device, HostDevice, Global, Grid_Global)

// CUDA data attributes; mutually exclusive
ENUM_CLASS(CUDADataAttr, Constant, Device, Managed, Pinned, Shared, Texture)

// Fortran names may have up to 63 characters (See Fortran 2018 C601).
static constexpr int maxNameLen{63};

// !DIR$ IGNORE_TKR [[(letters) name] ... letters
// "A" expands to all of TKRDM
ENUM_CLASS(IgnoreTKR,
    Type, // T - don't check type category
    Kind, // K - don't check kind
    Rank, // R - don't check ranks
    Device, // D - don't check host/device residence
    Managed, // M - don't check managed storage
    Contiguous) // C - legacy; disabled NVFORTRAN's convention that leading
                // dimension of assumed-shape was contiguous
using IgnoreTKRSet = EnumSet<IgnoreTKR, 8>;
// IGNORE_TKR(A) = IGNORE_TKR(TKRDM)
static constexpr IgnoreTKRSet ignoreTKRAll{IgnoreTKR::Type, IgnoreTKR::Kind,
    IgnoreTKR::Rank, IgnoreTKR::Device, IgnoreTKR::Managed};
std::string AsFortran(IgnoreTKRSet);

bool AreCompatibleCUDADataAttrs(
    std::optional<CUDADataAttr>, std::optional<CUDADataAttr>, IgnoreTKRSet);

} // namespace Fortran::common
#endif // FORTRAN_COMMON_FORTRAN_H_
