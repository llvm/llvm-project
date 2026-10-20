//==- BuiltinsARM.td - ARM Builtin function database -------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ARM-specific builtin function database.
//
//===----------------------------------------------------------------------===//

// The format of this database matches clang/Basic/Builtins.def

include "clang/Basic/BuiltinsARMBase.td"

// In libgcc
let Attributes = [RuntimeLibraryFunction] in {
	def __clear_cache : ARMNoPrefixBuiltin<"void (void *, void *)">;
}

// 16-bit multiplications
let Attributes = [NoThrow, Const] in {
	def smulbb : ARMBuiltin<"int (int, int)">;
	def smulbt : ARMBuiltin<"int (int, int)">;
	def smultb : ARMBuiltin<"int (int, int)">;
	def smultt : ARMBuiltin<"int (int, int)">;
	def smulwb : ARMBuiltin<"int (int, int)">;
	def smulwt : ARMBuiltin<"int (int, int)">;
}

// Saturating arithmetic
let Attributes = [NoThrow, Const] in {
	def qadd 	: ARMBuiltin<"int (int, int)">;
	def qsub 	: ARMBuiltin<"int (int, int)">;
	def qdbl 	: ARMBuiltin<"int (int)">;
	def ssat 	: ARMBuiltin<"int (int, unsigned int)">;
	def usat 	: ARMBuiltin<"unsigned int (int, unsigned int)">;
	def smlabb 	: ARMBuiltin<"int (int, int, int)">;
	def smlabt 	: ARMBuiltin<"int (int, int, int)">;
	def smlatb 	: ARMBuiltin<"int (int, int, int)">;
	def smlatt 	: ARMBuiltin<"int (int, int, int)">;
	def smlawb 	: ARMBuiltin<"int (int, int, int)">;
	def smlawt 	: ARMBuiltin<"int (int, int, int)">;
	def ssat16 	: ARMBuiltin<"int (int, int)">;
	def usat16 	: ARMBuiltin<"int (int, int)">;
	def sxtab16 : ARMBuiltin<"int (int, int)">;
	def sxtb16 	: ARMBuiltin<"int (int)">;
	def uxtab16 : ARMBuiltin<"int (int, int)">;
	def uxtb16 	: ARMBuiltin<"int (int)">;
	def sel 	: ARMBuiltin<"int (int, int)">;
	def qadd8 	: ARMBuiltin<"int (int, int)">;
	def qsub8 	: ARMBuiltin<"int (int, int)">;
	def sadd8 	: ARMBuiltin<"int (int, int)">;
	def shadd8 	: ARMBuiltin<"int (int, int)">;
	def shsub8 	: ARMBuiltin<"int (int, int)">;
	def ssub8 	: ARMBuiltin<"int (int, int)">;
	def uadd8 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uhadd8 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uhsub8 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uqadd8 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uqsub8 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def usub8 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
}

// Sum of 8-bit absolute differences
let Attributes = [NoThrow, Const] in {
	def usad8 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def usada8 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int, unsigned int)">;
}

// Parallel 16-bit addition and subtraction
let Attributes = [NoThrow, Const] in {
	def qadd16 	: ARMBuiltin<"int (int, int)">;
	def qasx 	: ARMBuiltin<"int (int, int)">;
	def qsax 	: ARMBuiltin<"int (int, int)">;
	def qsub16 	: ARMBuiltin<"int (int, int)">;
	def sadd16 	: ARMBuiltin<"int (int, int)">;
	def sasx 	: ARMBuiltin<"int (int, int)">;
	def shadd16 : ARMBuiltin<"int (int, int)">;
	def shasx 	: ARMBuiltin<"int (int, int)">;
	def shsax 	: ARMBuiltin<"int (int, int)">;
	def shsub16 : ARMBuiltin<"int (int, int)">;
	def ssax 	: ARMBuiltin<"int (int, int)">;
	def ssub16 	: ARMBuiltin<"int (int, int)">;
	def uadd16 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uasx 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uhadd16 : ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uhasx 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uhsax 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uhsub16 : ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uqadd16 : ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uqasx 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uqsax 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def uqsub16 : ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def usax 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def usub16 	: ARMBuiltin<"unsigned int (unsigned int, unsigned int)">;
}

// Parallel 16-bit multiplication
let Attributes = [NoThrow, Const] in {
	def smlad 	: ARMBuiltin<"int (int, int, int)">;
	def smladx 	: ARMBuiltin<"int (int, int, int)">;
	def smlald 	: ARMBuiltin<"long long int (int, int, long long int)">;
	def smlaldx : ARMBuiltin<"long long int (int, int, long long int)">;
	def smlsd 	: ARMBuiltin<"int (int, int, int)">;
	def smlsdx 	: ARMBuiltin<"int (int, int, int)">;
	def smlsld 	: ARMBuiltin<"long long int (int, int, long long int)">;
	def smlsldx : ARMBuiltin<"long long int (int, int, long long int)">;
	def smuad 	: ARMBuiltin<"int (int, int)">;
	def smuadx 	: ARMBuiltin<"int (int, int)">;
	def smusd 	: ARMBuiltin<"int (int, int)">;
	def smusdx 	: ARMBuiltin<"int (int, int)">;
}

// Bit manipulation
let Attributes = [NoThrow, Const] in {
	def rbit 	: ARMBuiltin<"unsigned int (unsigned int)">;
	def clz 	: ARMBuiltin<"unsigned int (uint32_t)">;
	def clz64 	: ARMBuiltin<"unsigned int (uint64_t)">;
	def cls 	: ARMBuiltin<"unsigned int (uint32_t)">;
	def cls64 	: ARMBuiltin<"unsigned int (uint64_t)">;
}

// Store and load exclusive
let Attributes = [CustomTypeChecking] in {
	def ldrexd 	: ARMBuiltin<"void (...)">;
	def strexd 	: ARMBuiltin<"int (...)">;
	def ldrex 	: ARMBuiltin<"void (...)">;
	def ldaex 	: ARMBuiltin<"void (...)">;
	def strex 	: ARMBuiltin<"int (...)">;
	def stlex 	: ARMBuiltin<"int (...)">;
}

def clrex : ARMBuiltin<"void ()">;

// VFP
let Attributes = [NoThrow, Const] in {
	def get_fpscr 	: ARMBuiltin<"unsigned int ()">;
	def set_fpscr 	: ARMBuiltin<"void (unsigned int)">;
	def vcvtr_f 	: ARMBuiltin<"float (float, int)">;
	def vcvtr_d 	: ARMBuiltin<"float (double, int)">;
}

// Coprocessor
def ldc 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, void const *)">;
def ldcl 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, void const *)">;
def ldc2 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, void const *)">;
def ldc2l 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, void const *)">;
def stc 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, void *)">;
def stcl 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, void *)">;
def stc2 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, void *)">;
def stc2l 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, void *)">;
def cdp 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
def cdp2 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
def mcr 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, unsigned int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
def mcr2 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, unsigned int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
def mrc 	: ARMBuiltin<"unsigned int (unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
def mrc2 	: ARMBuiltin<"unsigned int (unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
def mcrr 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, unsigned long long int, unsigned _Constant int)">;
def mcrr2 	: ARMBuiltin<"void (unsigned _Constant int, unsigned _Constant int, unsigned long long int, unsigned _Constant int)">;
def mrrc 	: ARMBuiltin<"unsigned long long int (unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
def mrrc2 	: ARMBuiltin<"unsigned long long int (unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;

// CRC32
let Attributes = [NoThrow, Const], Features = "crc" in {
	def crc32b 	: ARMTargetBuiltin<"unsigned int (unsigned int, unsigned char)">;
	def crc32cb : ARMTargetBuiltin<"unsigned int (unsigned int, unsigned char)">;
	def crc32h 	: ARMTargetBuiltin<"unsigned int (unsigned int, unsigned short)">;
	def crc32ch : ARMTargetBuiltin<"unsigned int (unsigned int, unsigned short)">;
	def crc32w 	: ARMTargetBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def crc32cw : ARMTargetBuiltin<"unsigned int (unsigned int, unsigned int)">;
	def crc32d 	: ARMTargetBuiltin<"unsigned int (unsigned int, unsigned long long int)">;
	def crc32cd : ARMTargetBuiltin<"unsigned int (unsigned int, unsigned long long int)">;
}

// ARMv8-M Security Extensions a.k.a CMSE
let Attributes = [NoThrow] in {
	def cmse_TT 	: ARMBuiltin<"unsigned int (void *)">;
	def cmse_TTT 	: ARMBuiltin<"unsigned int (void *)">;
	def cmse_TTA 	: ARMBuiltin<"unsigned int (void *)">;
	def cmse_TTAT 	: ARMBuiltin<"unsigned int (void *)">;
}

// HINT
def nop 	: ARMBuiltin<"void ()">;
def yield 	: ARMBuiltin<"void ()">;
def wfe 	: ARMBuiltin<"void ()">;
def wfi 	: ARMBuiltin<"void ()">;
def sev 	: ARMBuiltin<"void ()">;
def sevl 	: ARMBuiltin<"void ()">;
def dbg 	: ARMBuiltin<"void (unsigned int)">;

let Attributes = [RequireDeclaration], Header = "arm_acle.h", Languages = "ALL_LANGUAGES" in {
	def __yield : ARMNoPrefixTargetLibBuiltin<"void ()">;
	def __wfe 	: ARMNoPrefixTargetLibBuiltin<"void ()">;
	def __wfi 	: ARMNoPrefixTargetLibBuiltin<"void ()">;
	def __sev 	: ARMNoPrefixTargetLibBuiltin<"void ()">;
	def __sevl 	: ARMNoPrefixTargetLibBuiltin<"void ()">;
}

// Data barrier
let Attributes = [NoThrow, Const] in {
	def dmb : ARMBuiltin<"void (unsigned int)">;
	def dsb : ARMBuiltin<"void (unsigned int)">;
	def isb : ARMBuiltin<"void (unsigned int)">;
}

let Attributes = [NoThrow, Const, RequireDeclaration], Header = "arm_acle.h", Languages = "ALL_LANGUAGES" in {
	def __dmb : ARMNoPrefixTargetLibBuiltin<"void (unsigned int)">;
	def __dsb : ARMNoPrefixTargetLibBuiltin<"void (unsigned int)">;
	def __isb : ARMNoPrefixTargetLibBuiltin<"void (unsigned int)">;
}

// Prefetch
let Attributes = [NoThrow, Const] in {
	def prefetch : ARMBuiltin<"void (void const *, unsigned int, unsigned int)">;
}

// System registers (ACLE)
let Attributes = [NoThrow, Const] in {
	def rsr 	: ARMBuiltin<"unsigned int (char const *)">;
	def rsr64 	: ARMBuiltin<"unsigned long long int (char const *)">;
	def rsrp 	: ARMBuiltin<"void * (char const *)">;
	def wsr 	: ARMBuiltin<"void (char const *, unsigned int)">;
	def wsr64 	: ARMBuiltin<"void (char const *, unsigned long long int)">;
	def wsrp 	: ARMBuiltin<"void (char const *, void const *)">;
}

// Misc
let Attributes = [Const] in {
	def __builtin_sponentry : ARMNoPrefixBuiltin<"void * ()">;
}

// MSVC
let Languages = "ALL_MS_LANGUAGES" in {
	def __emit 					: ARMNoPrefixLangBuiltin<"void (_Constant unsigned int const)">;
	def __ldrexd 				: ARMNoPrefixLangBuiltin<"int64_t (int64_t const volatile *)">;
	def _MoveFromCoprocessor 	: ARMNoPrefixLangBuiltin<"unsigned int (unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
	def _MoveFromCoprocessor2 	: ARMNoPrefixLangBuiltin<"unsigned int (unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
	def _MoveToCoprocessor 		: ARMNoPrefixLangBuiltin<"void (unsigned int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
	def _MoveToCoprocessor2 	: ARMNoPrefixLangBuiltin<"void (unsigned int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int, unsigned _Constant int)">;
}

let Attributes = [NoThrow, RequireDeclaration], Header = "intrin.h", Languages = "ALL_MS_LANGUAGES" in {
	def _BitScanForward 					: ARMNoPrefixTargetLibBuiltin<"unsigned char (msuint32_t *, msuint32_t)">;
	def _BitScanReverse 					: ARMNoPrefixTargetLibBuiltin<"unsigned char (msuint32_t *, msuint32_t)">;
	def _BitScanForward64					: ARMNoPrefixTargetLibBuiltin<"unsigned char (msuint32_t *, unsigned long long int)">;
	def _BitScanReverse64					: ARMNoPrefixTargetLibBuiltin<"unsigned char (msuint32_t *, unsigned long long int)">;
	def _InterlockedAnd64					: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedDecrement64 			: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *)">;
	def _InterlockedExchange64 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedExchangeAdd64 			: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedExchangeSub64 			: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedIncrement64 			: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *)">;
	def _InterlockedOr64 					: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedXor64 					: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedExchangeAdd_acq 		: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedExchangeAdd_rel 		: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedExchangeAdd_nf 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedExchangeAdd8_acq 		: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedExchangeAdd8_rel 		: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedExchangeAdd8_nf 		: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedExchangeAdd16_acq 		: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedExchangeAdd16_rel 		: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedExchangeAdd16_nf 		: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedExchangeAdd64_acq 		: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedExchangeAdd64_rel 		: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedExchangeAdd64_nf 		: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedExchange8_acq 			: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedExchange8_nf 			: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedExchange8_rel 			: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedExchange16_acq 			: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedExchange16_nf 			: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedExchange16_rel 			: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedExchange_acq 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedExchange_nf 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedExchange_rel 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedExchange64_acq 			: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedExchange64_nf 			: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedExchange64_rel 			: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedExchangePointer_acq 	: ARMNoPrefixTargetLibBuiltin<"void * (void * volatile *, void *)">;
	def _InterlockedExchangePointer_nf 		: ARMNoPrefixTargetLibBuiltin<"void * (void * volatile *, void *)">;
	def _InterlockedExchangePointer_rel 	: ARMNoPrefixTargetLibBuiltin<"void * (void * volatile *, void *)">;
	def _InterlockedCompareExchange8_acq 	: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char, char)">;
	def _InterlockedCompareExchange8_nf 	: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char, char)">;
	def _InterlockedCompareExchange8_rel 	: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char, char)">;
	def _InterlockedCompareExchange16_acq 	: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short, short)">;
	def _InterlockedCompareExchange16_nf 	: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short, short)">;
	def _InterlockedCompareExchange16_rel 	: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short, short)">;
	def _InterlockedCompareExchange_acq 	: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t, msint32_t)">;
	def _InterlockedCompareExchange_nf 		: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t, msint32_t)">;
	def _InterlockedCompareExchange_rel 	: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t, msint32_t)">;
	def _InterlockedCompareExchange64_acq 	: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int, long long int)">;
	def _InterlockedCompareExchange64_nf 	: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int, long long int)">;
	def _InterlockedCompareExchange64_rel 	: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int, long long int)">;
	def _InterlockedCompareExchangePointer_acq : ARMNoPrefixTargetLibBuiltin<"void * (void * volatile *, void *, void *)">;
	def _InterlockedCompareExchangePointer_rel : ARMNoPrefixTargetLibBuiltin<"void * (void * volatile *, void *, void *)">;
	def _InterlockedOr8_acq 				: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedOr8_nf 					: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedOr8_rel 				: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedOr16_acq 				: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedOr16_nf 				: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedOr16_rel 				: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedOr_acq 					: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedOr_nf 					: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedOr_rel 					: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedOr64_acq 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedOr64_nf 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedOr64_rel 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedXor8_acq 				: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedXor8_nf 				: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedXor8_rel 				: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedXor16_acq 				: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedXor16_nf 				: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedXor16_rel				: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedXor_acq 				: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedXor_nf 					: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedXor_rel 				: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedXor64_acq 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedXor64_nf 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedXor64_rel 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedAnd8_acq 				: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedAnd8_nf 				: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedAnd8_rel 				: ARMNoPrefixTargetLibBuiltin<"char (char volatile *, char)">;
	def _InterlockedAnd16_acq 				: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedAnd16_nf 				: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedAnd16_rel 				: ARMNoPrefixTargetLibBuiltin<"short (short volatile *, short)">;
	def _InterlockedAnd_acq			 		: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedAnd_nf 					: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedAnd_rel 				: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *, msint32_t)">;
	def _InterlockedAnd64_acq 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedAnd64_nf 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedAnd64_rel 				: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *, long long int)">;
	def _InterlockedIncrement16_acq 		: ARMNoPrefixTargetLibBuiltin<"short (short volatile *)">;
	def _InterlockedIncrement16_nf 			: ARMNoPrefixTargetLibBuiltin<"short (short volatile *)">;
	def _InterlockedIncrement16_rel 		: ARMNoPrefixTargetLibBuiltin<"short (short volatile *)">;
	def _InterlockedIncrement_acq 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *)">;
	def _InterlockedIncrement_nf 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *)">;
	def _InterlockedIncrement_rel 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *)">;
	def _InterlockedIncrement64_acq 		: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *)">;
	def _InterlockedIncrement64_nf 			: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *)">;
	def _InterlockedIncrement64_rel 		: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *)">;
	def _InterlockedDecrement16_acq 		: ARMNoPrefixTargetLibBuiltin<"short (short volatile *)">;
	def _InterlockedDecrement16_nf 			: ARMNoPrefixTargetLibBuiltin<"short (short volatile *)">;
	def _InterlockedDecrement16_rel 		: ARMNoPrefixTargetLibBuiltin<"short (short volatile *)">;
	def _InterlockedDecrement_acq 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *)">;
	def _InterlockedDecrement_nf 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *)">;
	def _InterlockedDecrement_rel 			: ARMNoPrefixTargetLibBuiltin<"msint32_t (msint32_t volatile *)">;
	def _InterlockedDecrement64_acq 		: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *)">;
	def _InterlockedDecrement64_nf 			: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *)">;
	def _InterlockedDecrement64_rel 		: ARMNoPrefixTargetLibBuiltin<"long long int (long long int volatile *)">;
}

