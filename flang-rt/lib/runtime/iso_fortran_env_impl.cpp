//===-- lib/runtime/iso_fortran_env_impl.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides linkable symbols for the iso_fortran_env_impl module constants. It
// replaces the Fortran source (flang/module/iso_fortran_env_impl.f90) to remove
// the runtime library's build dependency on the Fortran compiler.
//
// The values here must stay in sync with the Fortran module source which
// generates the .mod file during the flang compiler build.
//
// Symbol naming follows Flang's module variable mangling:
//   _QM<module_name>EC<constant_name>
//
//===----------------------------------------------------------------------===//

#include "flang/Common/float128.h"
#include "flang/Common/float80.h"
#include "flang/Common/type-kinds.h"
#include <cstdint>

// Fortran merge(tsource, fsource, mask) to match the source module style.
static constexpr std::int32_t Merge(
    std::int32_t tsource, std::int32_t fsource, bool mask) {
  return mask ? tsource : fsource;
}

// Fortran digits() for integer kind K returns 8*K - 1 (excludes sign bit).
static constexpr int IntDigits(int kind) { return 8 * kind - 1; }

// Fortran digits() for unsigned kind K returns 8*K (no sign bit).
static constexpr int UintDigits(int kind) { return 8 * kind; }

// Fortran digits() for real kind K returns the binary precision.
static constexpr int RealDigits(int kind) {
  return Fortran::common::PrecisionOfRealKind(kind);
}

// The smallest valid real kind, used as a safe fallback when a selected
// real kind is unavailable (so digits() can be called without error).
static constexpr std::int32_t safeRealFallback{2};

// Integer kinds, selected -> safe -> validated.
static constexpr std::int32_t selectedInt8{1};
static constexpr std::int32_t selectedInt16{2};
static constexpr std::int32_t selectedInt32{4};
static constexpr std::int32_t selectedInt64{8};
static constexpr std::int32_t selectedInt128{16};

static constexpr std::int32_t safeInt8{
    Merge(selectedInt8, 1, selectedInt8 >= 0)};
static constexpr std::int32_t safeInt16{
    Merge(selectedInt16, 1, selectedInt16 >= 0)};
static constexpr std::int32_t safeInt32{
    Merge(selectedInt32, 1, selectedInt32 >= 0)};
static constexpr std::int32_t safeInt64{
    Merge(selectedInt64, 1, selectedInt64 >= 0)};
static constexpr std::int32_t safeInt128{
    Merge(selectedInt128, 1, selectedInt128 >= 0)};

static constexpr std::int32_t int8{Merge(
    selectedInt8, Merge(-2, -1, selectedInt8 >= 0), IntDigits(safeInt8) == 7)};
static constexpr std::int32_t int16{Merge(selectedInt16,
    Merge(-2, -1, selectedInt16 >= 0), IntDigits(safeInt16) == 15)};
static constexpr std::int32_t int32{Merge(selectedInt32,
    Merge(-2, -1, selectedInt32 >= 0), IntDigits(safeInt32) == 31)};
static constexpr std::int32_t int64{Merge(selectedInt64,
    Merge(-2, -1, selectedInt64 >= 0), IntDigits(safeInt64) == 63)};
static constexpr std::int32_t int128{Merge(selectedInt128,
    Merge(-2, -1, selectedInt128 >= 0), IntDigits(safeInt128) == 127)};

// Unsigned kinds, same selection as integer, validated with unsigned digits.
static constexpr std::int32_t selectedUInt8{selectedInt8};
static constexpr std::int32_t selectedUInt16{selectedInt16};
static constexpr std::int32_t selectedUInt32{selectedInt32};
static constexpr std::int32_t selectedUInt64{selectedInt64};
static constexpr std::int32_t selectedUInt128{selectedInt128};

static constexpr std::int32_t safeUInt8{safeInt8};
static constexpr std::int32_t safeUInt16{safeInt16};
static constexpr std::int32_t safeUInt32{safeInt32};
static constexpr std::int32_t safeUInt64{safeInt64};
static constexpr std::int32_t safeUInt128{safeInt128};

static constexpr std::int32_t uint8{Merge(selectedUInt8,
    Merge(-2, -1, selectedUInt8 >= 0), UintDigits(safeUInt8) == 8)};
static constexpr std::int32_t uint16{Merge(selectedUInt16,
    Merge(-2, -1, selectedUInt16 >= 0), UintDigits(safeUInt16) == 16)};
static constexpr std::int32_t uint32{Merge(selectedUInt32,
    Merge(-2, -1, selectedUInt32 >= 0), UintDigits(safeUInt32) == 32)};
static constexpr std::int32_t uint64{Merge(selectedUInt64,
    Merge(-2, -1, selectedUInt64 >= 0), UintDigits(safeUInt64) == 64)};
static constexpr std::int32_t uint128{Merge(selectedUInt128,
    Merge(-2, -1, selectedUInt128 >= 0), UintDigits(safeUInt128) == 128)};

// Logical kinds mirror integer kinds.
static constexpr std::int32_t logical8{int8};
static constexpr std::int32_t logical16{int16};
static constexpr std::int32_t logical32{int32};
static constexpr std::int32_t logical64{int64};

// Real kinds, selected -> safe -> validated.
static constexpr std::int32_t selectedReal16{2};
static constexpr std::int32_t selectedBfloat16{3};
static constexpr std::int32_t selectedReal32{4};
static constexpr std::int32_t selectedReal64{8};

#if FLANG_RT_SUPPORTS_REAL16
static constexpr std::int32_t selectedReal80{16};
#elif HAS_FLOAT80
static constexpr std::int32_t selectedReal80{10};
#else
static constexpr std::int32_t selectedReal80{-3};
#endif

#if FLANG_RT_SUPPORTS_REAL16
static constexpr std::int32_t selectedReal64x2{16};
static constexpr std::int32_t selectedReal128{16};
#elif HAS_FLOAT80
static constexpr std::int32_t selectedReal64x2{-1};
static constexpr std::int32_t selectedReal128{-1};
#else
static constexpr std::int32_t selectedReal64x2{-1};
static constexpr std::int32_t selectedReal128{-3};
#endif

static constexpr std::int32_t safeReal16{
    Merge(selectedReal16, safeRealFallback, selectedReal16 >= 0)};
static constexpr std::int32_t safeBfloat16{
    Merge(selectedBfloat16, safeRealFallback, selectedBfloat16 >= 0)};
static constexpr std::int32_t safeReal32{
    Merge(selectedReal32, safeRealFallback, selectedReal32 >= 0)};
static constexpr std::int32_t safeReal64{
    Merge(selectedReal64, safeRealFallback, selectedReal64 >= 0)};
static constexpr std::int32_t safeReal80{
    Merge(selectedReal80, safeRealFallback, selectedReal80 >= 0)};
static constexpr std::int32_t safeReal64x2{
    Merge(selectedReal64x2, safeRealFallback, selectedReal64x2 >= 0)};
static constexpr std::int32_t safeReal128{
    Merge(selectedReal128, safeRealFallback, selectedReal128 >= 0)};

static constexpr std::int32_t real16{Merge(selectedReal16,
    Merge(-2, -1, selectedReal16 >= 0), RealDigits(safeReal16) == 11)};
static constexpr std::int32_t bfloat16{Merge(selectedBfloat16,
    Merge(-2, -1, selectedBfloat16 >= 0), RealDigits(safeBfloat16) == 8)};
static constexpr std::int32_t real32{Merge(selectedReal32,
    Merge(-2, -1, selectedReal32 >= 0), RealDigits(safeReal32) == 24)};
static constexpr std::int32_t real64{Merge(selectedReal64,
    Merge(-2, -1, selectedReal64 >= 0), RealDigits(safeReal64) == 53)};
static constexpr std::int32_t real80{Merge(selectedReal80,
    Merge(-2, -1, selectedReal80 >= 0), RealDigits(safeReal80) == 64)};
static constexpr std::int32_t real64x2{Merge(selectedReal64x2,
    Merge(-2, -1, selectedReal64x2 >= 0), RealDigits(safeReal64x2) == 106)};
static constexpr std::int32_t real128{Merge(selectedReal128,
    Merge(-2, -1, selectedReal128 >= 0), RealDigits(safeReal128) == 113)};

// Exported symbols with Flang module-variable mangling.
#define FORTRAN_NAMED_CONST(name) _QMiso_fortran_env_implEC##name

extern "C" {

extern const std::int32_t FORTRAN_NAMED_CONST(selectedint8){selectedInt8};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedint16){selectedInt16};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedint32){selectedInt32};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedint64){selectedInt64};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedint128){selectedInt128};

extern const std::int32_t FORTRAN_NAMED_CONST(safeint8){safeInt8};
extern const std::int32_t FORTRAN_NAMED_CONST(safeint16){safeInt16};
extern const std::int32_t FORTRAN_NAMED_CONST(safeint32){safeInt32};
extern const std::int32_t FORTRAN_NAMED_CONST(safeint64){safeInt64};
extern const std::int32_t FORTRAN_NAMED_CONST(safeint128){safeInt128};

extern const std::int32_t FORTRAN_NAMED_CONST(int8){int8};
extern const std::int32_t FORTRAN_NAMED_CONST(int16){int16};
extern const std::int32_t FORTRAN_NAMED_CONST(int32){int32};
extern const std::int32_t FORTRAN_NAMED_CONST(int64){int64};
extern const std::int32_t FORTRAN_NAMED_CONST(int128){int128};

extern const std::int32_t FORTRAN_NAMED_CONST(selecteduint8){selectedUInt8};
extern const std::int32_t FORTRAN_NAMED_CONST(selecteduint16){selectedUInt16};
extern const std::int32_t FORTRAN_NAMED_CONST(selecteduint32){selectedUInt32};
extern const std::int32_t FORTRAN_NAMED_CONST(selecteduint64){selectedUInt64};
extern const std::int32_t FORTRAN_NAMED_CONST(selecteduint128){selectedUInt128};

extern const std::int32_t FORTRAN_NAMED_CONST(safeuint8){safeUInt8};
extern const std::int32_t FORTRAN_NAMED_CONST(safeuint16){safeUInt16};
extern const std::int32_t FORTRAN_NAMED_CONST(safeuint32){safeUInt32};
extern const std::int32_t FORTRAN_NAMED_CONST(safeuint64){safeUInt64};
extern const std::int32_t FORTRAN_NAMED_CONST(safeuint128){safeUInt128};

extern const std::int32_t FORTRAN_NAMED_CONST(uint8){uint8};
extern const std::int32_t FORTRAN_NAMED_CONST(uint16){uint16};
extern const std::int32_t FORTRAN_NAMED_CONST(uint32){uint32};
extern const std::int32_t FORTRAN_NAMED_CONST(uint64){uint64};
extern const std::int32_t FORTRAN_NAMED_CONST(uint128){uint128};

extern const std::int32_t FORTRAN_NAMED_CONST(logical8){logical8};
extern const std::int32_t FORTRAN_NAMED_CONST(logical16){logical16};
extern const std::int32_t FORTRAN_NAMED_CONST(logical32){logical32};
extern const std::int32_t FORTRAN_NAMED_CONST(logical64){logical64};

extern const std::int32_t FORTRAN_NAMED_CONST(selectedreal16){selectedReal16};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedbfloat16){
    selectedBfloat16};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedreal32){selectedReal32};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedreal64){selectedReal64};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedreal80){selectedReal80};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedreal64x2){
    selectedReal64x2};
extern const std::int32_t FORTRAN_NAMED_CONST(selectedreal128){selectedReal128};

extern const std::int32_t FORTRAN_NAMED_CONST(safereal16){safeReal16};
extern const std::int32_t FORTRAN_NAMED_CONST(safebfloat16){safeBfloat16};
extern const std::int32_t FORTRAN_NAMED_CONST(safereal32){safeReal32};
extern const std::int32_t FORTRAN_NAMED_CONST(safereal64){safeReal64};
extern const std::int32_t FORTRAN_NAMED_CONST(safereal80){safeReal80};
extern const std::int32_t FORTRAN_NAMED_CONST(safereal64x2){safeReal64x2};
extern const std::int32_t FORTRAN_NAMED_CONST(safereal128){safeReal128};

extern const std::int32_t FORTRAN_NAMED_CONST(real16){real16};
extern const std::int32_t FORTRAN_NAMED_CONST(bfloat16){bfloat16};
extern const std::int32_t FORTRAN_NAMED_CONST(real32){real32};
extern const std::int32_t FORTRAN_NAMED_CONST(real64){real64};
extern const std::int32_t FORTRAN_NAMED_CONST(real80){real80};
extern const std::int32_t FORTRAN_NAMED_CONST(real64x2){real64x2};
extern const std::int32_t FORTRAN_NAMED_CONST(real128){real128};

extern const std::int32_t FORTRAN_NAMED_CONST(
    __builtin_integer_kinds)[] = FORTRAN_INTEGER_KINDS;

extern const std::int32_t FORTRAN_NAMED_CONST(
    __builtin_logical_kinds)[] = FORTRAN_LOGICAL_KINDS;

// Target-filtered subset of FORTRAN_REAL_KINDS from type-kinds.h.
extern const std::int32_t FORTRAN_NAMED_CONST(__builtin_real_kinds)[]{
    2,
    3,
    4,
    8,
#if HAS_FLOAT80
    10,
#endif
#if FLANG_RT_SUPPORTS_REAL16
    16,
#endif
};

} // extern "C"
