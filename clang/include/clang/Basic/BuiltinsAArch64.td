//==- BuiltinsAArch64.td - AArch64 Builtin function database ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AArch64-specific builtin function database.  Users of
// this file must define the BUILTIN macro to make use of this information.
//
//===----------------------------------------------------------------------===//

// The format of this database matches clang/Basic/Builtins.def.

include "clang/Basic/BuiltinsAArch64Base.td"

// In libgcc
def __clear_cache : AArch64NoPrefixBuiltin<"void (void *, void *)">;

let Attributes = [CustomTypeChecking] in {
	def ldrex : AArch64Builtin<"void (...)">;
	def ldaex : AArch64Builtin<"void (...)">;
	def strex : AArch64Builtin<"int (...)">;
	def stlex : AArch64Builtin<"int (...)">;
}

def clrex : AArch64Builtin<"void ()">;

// Bit manipulation
let Attributes = [NoThrow, Const] in {
	def rbit : AArch64Builtin<"unsigned int (unsigned int)">;
	def rbit64 : AArch64Builtin<"uint64_t (uint64_t)">;
	def cls : AArch64Builtin<"unsigned int (uint32_t)">;
	def cls64 : AArch64Builtin<"unsigned int (uint64_t)">;
	def clz : AArch64Builtin<"unsigned int (uint32_t)">;
	def clz64 : AArch64Builtin<"unsigned int (uint64_t)">;
}

// HINT
def nop : AArch64Builtin<"void ()">;
def yield : AArch64Builtin<"void ()">;
def wfe : AArch64Builtin<"void ()">;
def wfi : AArch64Builtin<"void ()">;
def sev : AArch64Builtin<"void ()">;
def sevl : AArch64Builtin<"void ()">;
def chkfeat : AArch64Builtin<"uint64_t (uint64_t)">;

let Attributes = [RequireDeclaration], Languages = "ALL_LANGUAGES", Header = "arm_acle.h" in {
	def __yield : AArch64NoPrefixTargetLibBuiltin<"void ()">;
	def __wfe : AArch64NoPrefixTargetLibBuiltin<"void ()">;
	def __wfi : AArch64NoPrefixTargetLibBuiltin<"void ()">;
	def __sev : AArch64NoPrefixTargetLibBuiltin<"void ()">;
	def __sevl : AArch64NoPrefixTargetLibBuiltin<"void ()">;
}

// Like __builtin_trap but provide an 16-bit immediate reason code (which goes into `brk #N`).
let Attributes = [NoThrow, NoReturn] in {
	def trap : AArch64Builtin<"void (unsigned _Constant short)">;
}

// CRC32
let Attributes = [NoThrow, Const], Features = "crc" in {
	def crc32b : AArch64TargetBuiltin<"unsigned int (unsigned int, unsigned char)">;
	def crc32cb : AArch64TargetBuiltin<"unsigned int (unsigned int, unsigned char)">;
	def crc32h : AArch64TargetBuiltin<"unsigned int (unsigned int, unsigned short)">;
	def crc32ch : AArch64TargetBuiltin<"unsigned int (unsigned int, unsigned short)">;
	def crc32w : AArch64TargetBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def crc32cw : AArch64TargetBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def crc32d : AArch64TargetBuiltin<"unsigned int (unsigned int, uint64_t)">;
	def crc32cd : AArch64TargetBuiltin<"unsigned int (unsigned int, uint64_t)">;
}

// Memory Tagging Extensions (MTE)
let Attributes = [CustomTypeChecking], Features = "mte" in {
	def irg : AArch64TargetBuiltin<"void * (void *, unsigned int)">;
	def addg : AArch64TargetBuiltin<"void * (void *, unsigned int)">;
	def gmi : AArch64TargetBuiltin<"unsigned int (void *, unsigned int)">;
	def ldg : AArch64TargetBuiltin<"void * (void *)">;
	def stg : AArch64TargetBuiltin<"void (void *)">;
	def subp : AArch64TargetBuiltin<"unsigned int (void *, void *)">;
}

// SME state function
let Attributes = [NoThrow] in {
	def get_sme_state : AArch64Builtin<"void (uint64_t *, uint64_t *)">;
}

// Memory Operations
let Features = "mte,mops" in {
	def mops_memset_tag : AArch64TargetBuiltin<"void * (void *, int, size_t)">;
}

// Memory barrier
let Attributes = [NoThrow, Const] in {
	def dmb : AArch64Builtin<"void (unsigned int)">;
	def dsb : AArch64Builtin<"void (unsigned int)">;
	def isb : AArch64Builtin<"void (unsigned int)">;
}

let Attributes = [NoThrow, Const, RequireDeclaration], Languages = "ALL_LANGUAGES", Header = "arm_acle.h" in {
	def __dmb : AArch64NoPrefixTargetLibBuiltin<"void (unsigned int)">;
	def __dsb : AArch64NoPrefixTargetLibBuiltin<"void (unsigned int)">;
	def __isb : AArch64NoPrefixTargetLibBuiltin<"void (unsigned int)">;
}

let Attributes = [NoThrow, Const], Features = "v8.3a" in {
	def jcvt : AArch64TargetBuiltin<"int32_t (double)">;
}

// Prefetch
let Attributes = [NoThrow, Const] in {
	def prefetch : AArch64Builtin<"void (const void *, unsigned int, unsigned int, unsigned int, unsigned int)">;
}

// Range Prefetch
let Attributes = [NoThrow] in {
	def range_prefetch_x : AArch64Builtin<"void (const void *, unsigned int, unsigned int, int, unsigned int, int, size_t)">;
	def range_prefetch : AArch64Builtin<"void (const void *, unsigned int, unsigned int, uint64_t)">;
}

// System Registers
let Attributes = [NoThrow, Const] in {
	def rsr : AArch64Builtin<"unsigned int (const char *)">;
	def rsr64 : AArch64Builtin<"uint64_t (const char *)">;
	def rsrp : AArch64Builtin<"void * (const char *)">;
	def wsr : AArch64Builtin<"void (const char *, unsigned int)">;
	def wsr64 : AArch64Builtin<"void (const char *, uint64_t)">;
	def wsrp : AArch64Builtin<"void (const char *, const void *)">;
}

let Attributes = [NoThrow, Const], Features = "d128" in {
	def rsr128 : AArch64TargetBuiltin<"__uint128_t (const char *)">;
	def wsr128 : AArch64TargetBuiltin<"void (const char *, __uint128_t)">;
}

// Misc
let Attributes = [Const] in {
	def __builtin_sponentry : AArch64NoPrefixBuiltin<"void * ()">;
}

// Armv8.5-A FP rounding intrinsics
let Features = "v8.5a" in {
	def rint32zf : AArch64TargetBuiltin<"float (float)">;
	def rint32z : AArch64TargetBuiltin<"double (double)">;
	def rint64zf : AArch64TargetBuiltin<"float (float)">;
	def rint64z : AArch64TargetBuiltin<"double (double)">;
	def rint32xf : AArch64TargetBuiltin<"float (float)">;
	def rint32x : AArch64TargetBuiltin<"double (double)">;
	def rint64xf : AArch64TargetBuiltin<"float (float)">;
	def rint64x : AArch64TargetBuiltin<"double (double)">;
}

// Armv8.5-A Random number generation intrinsics
let Attributes = [NoThrow], Features = "rand" in {
	def rndr : AArch64TargetBuiltin<"int (uint64_t *)">;
	def rndrrs : AArch64TargetBuiltin<"int (uint64_t *)">;
}

// Armv8.7-A load/store 64-byte intrinsics
let Attributes = [NoThrow], Features = "ls64" in {
	def ld64b : AArch64TargetBuiltin<"void (const void *, uint64_t *)">;
	def st64b : AArch64TargetBuiltin<"void (void *, const uint64_t *)">;
	def st64bv : AArch64TargetBuiltin<"uint64_t (void *, const uint64_t *)">;
	def st64bv0 : AArch64TargetBuiltin<"uint64_t (void *, const uint64_t *)">;
}

// Armv9.3-A Guarded Control Stack
let Attributes = [NoThrow], Features = "gcs" in {
	def gcspopm : AArch64TargetBuiltin<"uint64_t (uint64_t)">;
	def gcsss : AArch64TargetBuiltin<"void * (void *)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _BitScanForward : AArch64NoPrefixTargetLibBuiltin<"unsigned char (msuint32_t *, msuint32_t)">;
	def _BitScanReverse : AArch64NoPrefixTargetLibBuiltin<"unsigned char (msuint32_t *, msuint32_t)">;
	def _BitScanForward64 : AArch64NoPrefixTargetLibBuiltin<"unsigned char (msuint32_t *, unsigned long long int)">;
	def _BitScanReverse64 : AArch64NoPrefixTargetLibBuiltin<"unsigned char (msuint32_t *, unsigned long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedAdd : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedAdd64 : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedAnd64 : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedDecrement64 : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *)">;
	def _InterlockedExchange64 : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedExchangeAdd64 : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedExchangeSub64 : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedIncrement64 : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *)">;
	def _InterlockedOr64 : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedXor64 : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedAdd_acq : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedAdd_rel : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedAdd_nf : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedAdd64_acq : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedAdd64_rel : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedAdd64_nf : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedExchangeAdd_acq : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedExchangeAdd_rel : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedExchangeAdd_nf : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedExchangeAdd8_acq : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedExchangeAdd8_rel : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedExchangeAdd8_nf : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedExchangeAdd16_acq : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedExchangeAdd16_rel : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedExchangeAdd16_nf : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedExchangeAdd64_acq : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedExchangeAdd64_rel : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedExchangeAdd64_nf : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedExchange8_acq : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedExchange8_nf : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedExchange8_rel : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedExchange16_acq : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedExchange16_nf : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedExchange16_rel : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedExchange_acq : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedExchange_nf : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedExchange_rel : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedExchange64_acq : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedExchange64_nf : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedExchange64_rel : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedExchangePointer_acq : AArch64NoPrefixTargetLibBuiltin<"void * (void * volatile *, void *)">;
	def _InterlockedExchangePointer_nf : AArch64NoPrefixTargetLibBuiltin<"void * (void * volatile *, void *)">;
	def _InterlockedExchangePointer_rel : AArch64NoPrefixTargetLibBuiltin<"void * (void * volatile *, void *)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedCompareExchange8_acq : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char, char)">;
	def _InterlockedCompareExchange8_nf : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char, char)">;
	def _InterlockedCompareExchange8_rel : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char, char)">;
	def _InterlockedCompareExchange16_acq : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short, short)">;
	def _InterlockedCompareExchange16_nf : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short, short)">;
	def _InterlockedCompareExchange16_rel : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short, short)">;
	def _InterlockedCompareExchange_acq : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t, msint32_t)">;
	def _InterlockedCompareExchange_nf : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t, msint32_t)">;
	def _InterlockedCompareExchange_rel : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t, msint32_t)">;
	def _InterlockedCompareExchange64_acq : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int, long long int)">;
	def _InterlockedCompareExchange64_nf : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int, long long int)">;
	def _InterlockedCompareExchange64_rel : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int, long long int)">;
	def _InterlockedCompareExchangePointer_acq : AArch64NoPrefixTargetLibBuiltin<"void * (void * volatile *, void *, void *)">;
	def _InterlockedCompareExchangePointer_rel : AArch64NoPrefixTargetLibBuiltin<"void * (void * volatile *, void *, void *)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedCompareExchange128 : AArch64NoPrefixTargetLibBuiltin<"unsigned char (volatile long long int *, long long int, long long int, long long int *)">;
	def _InterlockedCompareExchange128_acq : AArch64NoPrefixTargetLibBuiltin<"unsigned char (volatile long long int *, long long int, long long int, long long int *)">;
	def _InterlockedCompareExchange128_nf  : AArch64NoPrefixTargetLibBuiltin<"unsigned char (volatile long long int *, long long int, long long int, long long int *)">;
	def _InterlockedCompareExchange128_rel : AArch64NoPrefixTargetLibBuiltin<"unsigned char (volatile long long int *, long long int, long long int, long long int *)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedOr8_acq : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedOr8_nf : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedOr8_rel : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedOr16_acq : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedOr16_nf : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedOr16_rel : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedOr_acq : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedOr_nf : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedOr_rel : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedOr64_acq : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedOr64_nf : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedOr64_rel : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedXor8_acq : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedXor8_nf : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedXor8_rel : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedXor16_acq : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedXor16_nf : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedXor16_rel : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedXor_acq : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedXor_nf : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedXor_rel : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedXor64_acq : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedXor64_nf : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedXor64_rel : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedAnd8_acq : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedAnd8_nf : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedAnd8_rel : AArch64NoPrefixTargetLibBuiltin<"char (volatile char *, char)">;
	def _InterlockedAnd16_acq : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedAnd16_nf : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedAnd16_rel : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *, short)">;
	def _InterlockedAnd_acq : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedAnd_nf : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedAnd_rel : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *, msint32_t)">;
	def _InterlockedAnd64_acq : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedAnd64_nf : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
	def _InterlockedAnd64_rel : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *, long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedIncrement16_acq : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *)">;
	def _InterlockedIncrement16_nf : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *)">;
	def _InterlockedIncrement16_rel : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *)">;
	def _InterlockedIncrement_acq : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *)">;
	def _InterlockedIncrement_nf : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *)">;
	def _InterlockedIncrement_rel : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *)">;
	def _InterlockedIncrement64_acq : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *)">;
	def _InterlockedIncrement64_nf : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *)">;
	def _InterlockedIncrement64_rel : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _InterlockedDecrement16_acq : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *)">;
	def _InterlockedDecrement16_nf : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *)">;
	def _InterlockedDecrement16_rel : AArch64NoPrefixTargetLibBuiltin<"short (volatile short *)">;
	def _InterlockedDecrement_acq : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *)">;
	def _InterlockedDecrement_nf : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *)">;
	def _InterlockedDecrement_rel : AArch64NoPrefixTargetLibBuiltin<"msint32_t (volatile msint32_t *)">;
	def _InterlockedDecrement64_acq : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *)">;
	def _InterlockedDecrement64_nf : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *)">;
	def _InterlockedDecrement64_rel : AArch64NoPrefixTargetLibBuiltin<"long long int (volatile long long int *)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _ReadWriteBarrier : AArch64NoPrefixTargetLibBuiltin<"void ()">;
	def __getReg : AArch64NoPrefixTargetLibBuiltin<"unsigned long long int (int)">;
	def _ReadStatusReg : AArch64NoPrefixTargetLibBuiltin<"long long int (int)">;
	def _WriteStatusReg : AArch64NoPrefixTargetLibBuiltin<"void (int, long long int)">;
	def __sys : AArch64NoPrefixTargetLibBuiltin<"unsigned int (int, long long int)">;
	def _AddressOfReturnAddress : AArch64NoPrefixTargetLibBuiltin<"void * ()">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def __mulh : AArch64NoPrefixTargetLibBuiltin<"signed long long int (signed long long int, signed long long int)">;
	def __umulh : AArch64NoPrefixTargetLibBuiltin<"unsigned long long int (unsigned long long int, unsigned long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def __break : AArch64NoPrefixTargetLibBuiltin<"void (int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def __writex18byte : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t, unsigned char)">;
	def __writex18word : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t, unsigned short)">;
	def __writex18dword : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t, msuint32_t)">;
	def __writex18qword : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t, unsigned long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def __readx18byte : AArch64NoPrefixTargetLibBuiltin<"unsigned char (msuint32_t)">;
	def __readx18word : AArch64NoPrefixTargetLibBuiltin<"unsigned short (msuint32_t)">;
	def __readx18dword : AArch64NoPrefixTargetLibBuiltin<"msuint32_t (msuint32_t)">;
	def __readx18qword : AArch64NoPrefixTargetLibBuiltin<"unsigned long long int (msuint32_t)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def __addx18byte : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t, unsigned char)">;
	def __addx18word : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t, unsigned short)">;
	def __addx18dword : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t, msuint32_t)">;
	def __addx18qword : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t, unsigned long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def __incx18byte : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t)">;
	def __incx18word : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t)">;
	def __incx18dword : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t)">;
	def __incx18qword : AArch64NoPrefixTargetLibBuiltin<"void (msuint32_t)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _CopyDoubleFromInt64 : AArch64NoPrefixTargetLibBuiltin<"double (signed long long int)">;
	def _CopyFloatFromInt32 : AArch64NoPrefixTargetLibBuiltin<"float (signed int)">;
	def _CopyInt32FromFloat : AArch64NoPrefixTargetLibBuiltin<"signed int (float)">;
	def _CopyInt64FromDouble : AArch64NoPrefixTargetLibBuiltin<"signed long long int (double)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def _CountLeadingOnes : AArch64NoPrefixTargetLibBuiltin<"unsigned int (msuint32_t)">;
	def _CountLeadingOnes64 : AArch64NoPrefixTargetLibBuiltin<"unsigned int (unsigned long long int)">;
	def _CountLeadingSigns : AArch64NoPrefixTargetLibBuiltin<"unsigned int (signed msint32_t)">;
	def _CountLeadingSigns64 : AArch64NoPrefixTargetLibBuiltin<"unsigned int (signed long long int)">;
	def _CountLeadingZeros : AArch64NoPrefixTargetLibBuiltin<"unsigned int (msuint32_t)">;
	def _CountLeadingZeros64 : AArch64NoPrefixTargetLibBuiltin<"unsigned int (unsigned long long int)">;
	def _CountOneBits : AArch64NoPrefixTargetLibBuiltin<"unsigned int (msuint32_t)">;
	def _CountOneBits64 : AArch64NoPrefixTargetLibBuiltin<"unsigned int (unsigned long long int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def __prefetch : AArch64NoPrefixTargetLibBuiltin<"void (const void *)">;
}

let Attributes = [NoThrow, RequireDeclaration], Languages = "ALL_MS_LANGUAGES", Header = "intrin.h" in {
	def __hlt : AArch64NoPrefixTargetLibBuiltin<"unsigned int (unsigned int, ...)">;
}
