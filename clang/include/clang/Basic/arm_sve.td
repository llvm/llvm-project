//===--- arm_sve.td - ARM SVE compiler interface ------------------------===//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen definitions from which the ARM SVE header
//  file will be generated.  See:
//
//      https://developer.arm.com/architectures/system-architectures/software-standards/acle
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction definitions
//===----------------------------------------------------------------------===//
// Every intrinsic subclasses "Inst". An intrinsic has a name, a prototype and
// a sequence of typespecs.
//
// The name is the base name of the intrinsic, for example "svld1". This is
// then mangled by the tblgen backend to add type information ("svld1_s16").
//
// A typespec is a sequence of uppercase characters (modifiers) followed by one
// lowercase character. A typespec encodes a particular "base type" of the
// intrinsic.
//
// An example typespec is "Us" - unsigned short - svuint16_t. The available
// typespec codes are given below.
//
// The string given to an Inst class is a sequence of typespecs. The intrinsic
// is instantiated for every typespec in the sequence. For example "sdUsUd".
//
// The prototype is a string that defines the return type of the intrinsic
// and the type of each argument. The return type and every argument gets a
// "modifier" that can change in some way the "base type" of the intrinsic.
//
// The modifier 'd' means "default" and does not modify the base type in any
// way. The available modifiers are given below.
//
// Typespecs
// ---------
// c: char
// s: short
// i: int
// l: long
// f: float
// h: half-float
// d: double

// Typespec modifiers
// ------------------
// P: boolean
// U: unsigned

// Prototype modifiers
// -------------------
// prototype: return (arg, arg, ...)
//
// u: vector of unsigned integers
// d: default
// c: const pointer type
// P: predicate type
// e: 1/2 width unsigned elements, 2x element count
// h: 1/2 width elements, 2x element count
// q: 1/4 width elements, 4x element count
// o: 4x width elements, 1/4 element count
//
// i: constant uint64_t
//
// I: Predicate Pattern (sv_pattern)

// l: int64_t

// A: pointer to int8_t
// B: pointer to int16_t
// C: pointer to int32_t
// D: pointer to int64_t

// E: pointer to uint8_t
// F: pointer to uint16_t
// G: pointer to uint32_t
// H: pointer to uint64_t

// S: const pointer to int8_t
// T: const pointer to int16_t
// U: const pointer to int32_t
// V: const pointer to int64_t
//
// W: const pointer to uint8_t
// X: const pointer to uint16_t
// Y: const pointer to uint32_t
// Z: const pointer to uint64_t

class MergeType<int val, string suffix=""> {
  int Value = val;
  string Suffix = suffix;
}
def MergeNone    : MergeType<0>;
def MergeAny     : MergeType<1, "_x">;
def MergeOp1     : MergeType<2, "_m">;
def MergeZero    : MergeType<3, "_z">;
def MergeAnyExp  : MergeType<4, "_x">; // Use merged builtin with explicit
def MergeZeroExp : MergeType<5, "_z">; // generation of its inactive argument.

class EltType<int val> {
  int Value = val;
}
def EltTyInvalid : EltType<0>;
def EltTyInt8    : EltType<1>;
def EltTyInt16   : EltType<2>;
def EltTyInt32   : EltType<3>;
def EltTyInt64   : EltType<4>;
def EltTyFloat16 : EltType<5>;
def EltTyFloat32 : EltType<6>;
def EltTyFloat64 : EltType<7>;
def EltTyBool8   : EltType<8>;
def EltTyBool16  : EltType<9>;
def EltTyBool32  : EltType<10>;
def EltTyBool64  : EltType<11>;

class MemEltType<int val> {
  int Value = val;
}
def MemEltTyDefault   : MemEltType<0>;
def MemEltTyInt8      : MemEltType<1>;
def MemEltTyInt16     : MemEltType<2>;
def MemEltTyInt32     : MemEltType<3>;
def MemEltTyInt64     : MemEltType<4>;

class FlagType<int val> {
  int Value = val;
}

// These must be kept in sync with the flags in utils/TableGen/SveEmitter.h
// and include/clang/Basic/TargetBuiltins.h
def NoFlags                   : FlagType<0x00000000>;
def FirstEltType              : FlagType<0x00000001>;
//      :                                     :
//      :                                     :
def EltTypeMask               : FlagType<0x0000000f>;
def FirstMemEltType           : FlagType<0x00000010>;
//      :                                     :
//      :                                     :
def MemEltTypeMask            : FlagType<0x00000070>;
def FirstMergeTypeMask        : FlagType<0x00000080>;
//      :                                     :
//      :                                     :
def MergeTypeMask             : FlagType<0x00000380>;
def IsLoad                    : FlagType<0x00002000>;
def IsStore                   : FlagType<0x00004000>;
def IsGatherLoad              : FlagType<0x00008000>;
def IsScatterStore            : FlagType<0x00010000>;
def IsStructLoad              : FlagType<0x00020000>;
def IsStructStore             : FlagType<0x00040000>;
def IsZExtReturn              : FlagType<0x00080000>; // Return value is sign-extend by default

// These must be kept in sync with the flags in include/clang/Basic/TargetBuiltins.h
class ImmCheckType<int val> {
  int Value = val;
}
def ImmCheck0_31                : ImmCheckType<0>;  // 0..31 (used for e.g. predicate patterns)
def ImmCheck1_16                : ImmCheckType<1>;  // 1..16
def ImmCheckExtract             : ImmCheckType<2>;  // 0..(2048/sizeinbits(elt) - 1)
def ImmCheckShiftRight          : ImmCheckType<3>;  // 1..sizeinbits(elt)
def ImmCheckShiftRightNarrow    : ImmCheckType<4>;  // 1..sizeinbits(elt)/2
def ImmCheckShiftLeft           : ImmCheckType<5>;  // 0..(sizeinbits(elt) - 1)
def ImmCheck0_7                 : ImmCheckType<6>;  // 0..7
def ImmCheckLaneIndex           : ImmCheckType<7>;  // 0..(128/(1*sizeinbits(elt)) - 1)
def ImmCheckLaneIndexCompRotate : ImmCheckType<8>;  // 0..(128/(2*sizeinbits(elt)) - 1)
def ImmCheckLaneIndexDot        : ImmCheckType<9>;  // 0..(128/(4*sizeinbits(elt)) - 1)
def ImmCheckComplexRot90_270    : ImmCheckType<10>; // [90,270]
def ImmCheckComplexRotAll90     : ImmCheckType<11>; // [0, 90, 180,270]

class ImmCheck<int arg, ImmCheckType kind, int eltSizeArg = -1> {
  int Arg = arg;
  int EltSizeArg = eltSizeArg;
  ImmCheckType Kind = kind;
}

class Inst<string n, string p, string t, MergeType mt, string i,
           list<FlagType> ft, list<ImmCheck> ch, MemEltType met> {
  string Name = n;
  string Prototype = p;
  string Types = t;
  string ArchGuard = "";
  int Merge = mt.Value;
  string MergeSuffix = mt.Suffix;
  string LLVMIntrinsic = i;
  list<FlagType> Flags = ft;
  list<ImmCheck> ImmChecks = ch;
  int MemEltType = met.Value;
}

// SInst: Instruction with signed/unsigned suffix (e.g., "s8", "u8")
class SInst<string n, string p, string t, MergeType mt, string i = "",
            list<FlagType> ft = [], list<ImmCheck> ch = []>
    : Inst<n, p, t, mt, i, ft, ch, MemEltTyDefault> {
}

// MInst: Instructions which access memory
class MInst<string n, string p, string t, list<FlagType> f,
            MemEltType met = MemEltTyDefault, string i = "">
    : Inst<n, p, t, MergeNone, i, f, [], met> {
}

////////////////////////////////////////////////////////////////////////////////
// Loads

// Load one vector (scalar base)
def SVLD1   : MInst<"svld1[_{2}]", "dPc", "csilUcUsUiUlhfd", [IsLoad]>;
def SVLD1SB : MInst<"svld1sb_{d}", "dPS", "silUsUiUl",       [IsLoad],               MemEltTyInt8>;
def SVLD1UB : MInst<"svld1ub_{d}", "dPW", "silUsUiUl",       [IsLoad, IsZExtReturn], MemEltTyInt8>;
def SVLD1SH : MInst<"svld1sh_{d}", "dPT", "ilUiUl",          [IsLoad],               MemEltTyInt16>;
def SVLD1UH : MInst<"svld1uh_{d}", "dPX", "ilUiUl",          [IsLoad, IsZExtReturn], MemEltTyInt16>;
def SVLD1SW : MInst<"svld1sw_{d}", "dPU", "lUl",             [IsLoad],               MemEltTyInt32>;
def SVLD1UW : MInst<"svld1uw_{d}", "dPY", "lUl",             [IsLoad, IsZExtReturn], MemEltTyInt32>;

// Load one vector (scalar base, VL displacement)
def SVLD1_VNUM   : MInst<"svld1_vnum[_{2}]", "dPcl", "csilUcUsUiUlhfd", [IsLoad]>;
def SVLD1SB_VNUM : MInst<"svld1sb_vnum_{d}", "dPSl", "silUsUiUl",       [IsLoad],               MemEltTyInt8>;
def SVLD1UB_VNUM : MInst<"svld1ub_vnum_{d}", "dPWl", "silUsUiUl",       [IsLoad, IsZExtReturn], MemEltTyInt8>;
def SVLD1SH_VNUM : MInst<"svld1sh_vnum_{d}", "dPTl", "ilUiUl",          [IsLoad],               MemEltTyInt16>;
def SVLD1UH_VNUM : MInst<"svld1uh_vnum_{d}", "dPXl", "ilUiUl",          [IsLoad, IsZExtReturn], MemEltTyInt16>;
def SVLD1SW_VNUM : MInst<"svld1sw_vnum_{d}", "dPUl", "lUl",             [IsLoad],               MemEltTyInt32>;
def SVLD1UW_VNUM : MInst<"svld1uw_vnum_{d}", "dPYl", "lUl",             [IsLoad, IsZExtReturn], MemEltTyInt32>;

// First-faulting load one vector (scalar base)
def SVLDFF1   : MInst<"svldff1[_{2}]", "dPc", "csilUcUsUiUlhfd", [IsLoad],               MemEltTyDefault, "aarch64_sve_ldff1">;
def SVLDFF1SB : MInst<"svldff1sb_{d}", "dPS", "silUsUiUl",       [IsLoad],               MemEltTyInt8,    "aarch64_sve_ldff1">;
def SVLDFF1UB : MInst<"svldff1ub_{d}", "dPW", "silUsUiUl",       [IsLoad, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldff1">;
def SVLDFF1SH : MInst<"svldff1sh_{d}", "dPT", "ilUiUl",          [IsLoad],               MemEltTyInt16,   "aarch64_sve_ldff1">;
def SVLDFF1UH : MInst<"svldff1uh_{d}", "dPX", "ilUiUl",          [IsLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1">;
def SVLDFF1SW : MInst<"svldff1sw_{d}", "dPU", "lUl",             [IsLoad],               MemEltTyInt32,   "aarch64_sve_ldff1">;
def SVLDFF1UW : MInst<"svldff1uw_{d}", "dPY", "lUl",             [IsLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldff1">;

// First-faulting load one vector (scalar base, VL displacement)
def SVLDFF1_VNUM   : MInst<"svldff1_vnum[_{2}]", "dPcl", "csilUcUsUiUlhfd", [IsLoad],               MemEltTyDefault, "aarch64_sve_ldff1">;
def SVLDFF1SB_VNUM : MInst<"svldff1sb_vnum_{d}", "dPSl", "silUsUiUl",       [IsLoad],               MemEltTyInt8,    "aarch64_sve_ldff1">;
def SVLDFF1UB_VNUM : MInst<"svldff1ub_vnum_{d}", "dPWl", "silUsUiUl",       [IsLoad, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldff1">;
def SVLDFF1SH_VNUM : MInst<"svldff1sh_vnum_{d}", "dPTl", "ilUiUl",          [IsLoad],               MemEltTyInt16,   "aarch64_sve_ldff1">;
def SVLDFF1UH_VNUM : MInst<"svldff1uh_vnum_{d}", "dPXl", "ilUiUl",          [IsLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1">;
def SVLDFF1SW_VNUM : MInst<"svldff1sw_vnum_{d}", "dPUl", "lUl",             [IsLoad],               MemEltTyInt32,   "aarch64_sve_ldff1">;
def SVLDFF1UW_VNUM : MInst<"svldff1uw_vnum_{d}", "dPYl", "lUl",             [IsLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldff1">;

// Non-faulting load one vector (scalar base)
def SVLDNF1   : MInst<"svldnf1[_{2}]", "dPc", "csilUcUsUiUlhfd", [IsLoad],               MemEltTyDefault, "aarch64_sve_ldnf1">;
def SVLDNF1SB : MInst<"svldnf1sb_{d}", "dPS", "silUsUiUl",       [IsLoad],               MemEltTyInt8,    "aarch64_sve_ldnf1">;
def SVLDNF1UB : MInst<"svldnf1ub_{d}", "dPW", "silUsUiUl",       [IsLoad, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldnf1">;
def SVLDNF1SH : MInst<"svldnf1sh_{d}", "dPT", "ilUiUl",          [IsLoad],               MemEltTyInt16,   "aarch64_sve_ldnf1">;
def SVLDNF1UH : MInst<"svldnf1uh_{d}", "dPX", "ilUiUl",          [IsLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnf1">;
def SVLDNF1SW : MInst<"svldnf1sw_{d}", "dPU", "lUl",             [IsLoad],               MemEltTyInt32,   "aarch64_sve_ldnf1">;
def SVLDNF1UW : MInst<"svldnf1uw_{d}", "dPY", "lUl",             [IsLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldnf1">;

// Non-faulting load one vector (scalar base, VL displacement)
def SVLDNF1_VNUM   : MInst<"svldnf1_vnum[_{2}]", "dPcl", "csilUcUsUiUlhfd", [IsLoad],               MemEltTyDefault, "aarch64_sve_ldnf1">;
def SVLDNF1SB_VNUM : MInst<"svldnf1sb_vnum_{d}", "dPSl", "silUsUiUl",       [IsLoad],               MemEltTyInt8,    "aarch64_sve_ldnf1">;
def SVLDNF1UB_VNUM : MInst<"svldnf1ub_vnum_{d}", "dPWl", "silUsUiUl",       [IsLoad, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldnf1">;
def SVLDNF1SH_VNUM : MInst<"svldnf1sh_vnum_{d}", "dPTl", "ilUiUl",          [IsLoad],               MemEltTyInt16,   "aarch64_sve_ldnf1">;
def SVLDNF1UH_VNUM : MInst<"svldnf1uh_vnum_{d}", "dPXl", "ilUiUl",          [IsLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnf1">;
def SVLDNF1SW_VNUM : MInst<"svldnf1sw_vnum_{d}", "dPUl", "lUl",             [IsLoad],               MemEltTyInt32,   "aarch64_sve_ldnf1">;
def SVLDNF1UW_VNUM : MInst<"svldnf1uw_vnum_{d}", "dPYl", "lUl",             [IsLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldnf1">;

// Load one vector, unextended load, non-temporal (scalar base)
def SVLDNT1 : MInst<"svldnt1[_{2}]", "dPc", "csilUcUsUiUlhfd", [IsLoad], MemEltTyDefault, "aarch64_sve_ldnt1">;

// Load one vector, unextended load, non-temporal (scalar base, VL displacement)
def SVLDNT1_VNUM : MInst<"svldnt1_vnum[_{2}]", "dPcl", "csilUcUsUiUlhfd", [IsLoad], MemEltTyDefault, "aarch64_sve_ldnt1">;


////////////////////////////////////////////////////////////////////////////////
// Stores

// Store one vector (scalar base)
def SVST1    : MInst<"svst1[_{d}]",  "vPpd", "csilUcUsUiUlhfd", [IsStore]>;
def SVST1B_S : MInst<"svst1b[_{d}]", "vPAd", "sil",             [IsStore], MemEltTyInt8>;
def SVST1B_U : MInst<"svst1b[_{d}]", "vPEd", "UsUiUl",          [IsStore], MemEltTyInt8>;
def SVST1H_S : MInst<"svst1h[_{d}]", "vPBd", "il",              [IsStore], MemEltTyInt16>;
def SVST1H_U : MInst<"svst1h[_{d}]", "vPFd", "UiUl",            [IsStore], MemEltTyInt16>;
def SVST1W_S : MInst<"svst1w[_{d}]", "vPCd", "l",               [IsStore], MemEltTyInt32>;
def SVST1W_U : MInst<"svst1w[_{d}]", "vPGd", "Ul",              [IsStore], MemEltTyInt32>;

// Store one vector (scalar base, VL displacement)
def SVST1_VNUM    : MInst<"svst1_vnum[_{d}]",  "vPpld", "csilUcUsUiUlhfd", [IsStore]>;
def SVST1B_VNUM_S : MInst<"svst1b_vnum[_{d}]", "vPAld", "sil",             [IsStore], MemEltTyInt8>;
def SVST1B_VNUM_U : MInst<"svst1b_vnum[_{d}]", "vPEld", "UsUiUl",          [IsStore], MemEltTyInt8>;
def SVST1H_VNUM_S : MInst<"svst1h_vnum[_{d}]", "vPBld", "il",              [IsStore], MemEltTyInt16>;
def SVST1H_VNUM_U : MInst<"svst1h_vnum[_{d}]", "vPFld", "UiUl",            [IsStore], MemEltTyInt16>;
def SVST1W_VNUM_S : MInst<"svst1w_vnum[_{d}]", "vPCld", "l",               [IsStore], MemEltTyInt32>;
def SVST1W_VNUM_U : MInst<"svst1w_vnum[_{d}]", "vPGld", "Ul",              [IsStore], MemEltTyInt32>;

// Store one vector, with no truncation, non-temporal (scalar base)
def SVSTNT1 : MInst<"svstnt1[_{d}]", "vPpd", "csilUcUsUiUlhfd", [IsStore], MemEltTyDefault, "aarch64_sve_stnt1">;

// Store one vector, with no truncation, non-temporal (scalar base, VL displacement)
def SVSTNT1_VNUM : MInst<"svstnt1_vnum[_{d}]", "vPpld", "csilUcUsUiUlhfd", [IsStore], MemEltTyDefault, "aarch64_sve_stnt1">;

////////////////////////////////////////////////////////////////////////////////
// Integer arithmetic

multiclass SInstZPZ<string name, string types, string intrinsic, list<FlagType> flags=[]> {
  def _M : SInst<name # "[_{d}]", "ddPd", types, MergeOp1,     intrinsic, flags>;
  def _X : SInst<name # "[_{d}]", "dPd",  types, MergeAnyExp,  intrinsic, flags>;
  def _Z : SInst<name # "[_{d}]", "dPd",  types, MergeZeroExp, intrinsic, flags>;
}

defm SVABS : SInstZPZ<"svabs", "csil", "aarch64_sve_abs">;
defm SVNEG : SInstZPZ<"svneg", "csil", "aarch64_sve_neg">;

//------------------------------------------------------------------------------

multiclass SInstZPZZ<string name, string types, string intrinsic, list<FlagType> flags=[]> {
  def _M   : SInst<name # "[_{d}]",   "dPdd", types, MergeOp1,  intrinsic, flags>;
  def _X   : SInst<name # "[_{d}]",   "dPdd", types, MergeAny,  intrinsic, flags>;
  def _Z   : SInst<name # "[_{d}]",   "dPdd", types, MergeZero, intrinsic, flags>;
}

defm SVABD_S  : SInstZPZZ<"svabd",  "csil",         "aarch64_sve_sabd">;
defm SVABD_U  : SInstZPZZ<"svabd",  "UcUsUiUl",     "aarch64_sve_uabd">;
defm SVADD    : SInstZPZZ<"svadd",  "csilUcUsUiUl", "aarch64_sve_add">;
defm SVDIV_S  : SInstZPZZ<"svdiv",  "il",           "aarch64_sve_sdiv">;
defm SVDIV_U  : SInstZPZZ<"svdiv",  "UiUl",         "aarch64_sve_udiv">;
defm SVDIVR_S : SInstZPZZ<"svdivr", "il",           "aarch64_sve_sdivr">;
defm SVDIVR_U : SInstZPZZ<"svdivr", "UiUl",         "aarch64_sve_udivr">;
defm SVMAX_S  : SInstZPZZ<"svmax",  "csil",         "aarch64_sve_smax">;
defm SVMAX_U  : SInstZPZZ<"svmax",  "UcUsUiUl",     "aarch64_sve_umax">;
defm SVMIN_S  : SInstZPZZ<"svmin",  "csil",         "aarch64_sve_smin">;
defm SVMIN_U  : SInstZPZZ<"svmin",  "UcUsUiUl",     "aarch64_sve_umin">;
defm SVMUL    : SInstZPZZ<"svmul",  "csilUcUsUiUl", "aarch64_sve_mul">;
defm SVMULH_S : SInstZPZZ<"svmulh", "csil",         "aarch64_sve_smulh">;
defm SVMULH_U : SInstZPZZ<"svmulh", "UcUsUiUl",     "aarch64_sve_umulh">;
defm SVSUB    : SInstZPZZ<"svsub",  "csilUcUsUiUl", "aarch64_sve_sub">;
defm SVSUBR   : SInstZPZZ<"svsubr", "csilUcUsUiUl", "aarch64_sve_subr">;
////////////////////////////////////////////////////////////////////////////////
// Permutations and selection
def SVEXT        : SInst<"svext[_{d}]",       "dddi", "csilUcUsUiUlhfd", MergeNone, "aarch64_sve_ext", [], [ImmCheck<2, ImmCheckExtract, 1>]>;

////////////////////////////////////////////////////////////////////////////////
// Shifts
def SVASRD_M : SInst<"svasrd[_n_{d}]", "dPdi", "csil",            MergeOp1,  "aarch64_sve_asrd", [], [ImmCheck<2, ImmCheckShiftRight, 1>]>;

////////////////////////////////////////////////////////////////////////////////
// SVE2 - Narrowing DSP operations
let ArchGuard = "defined(__ARM_FEATURE_SVE2)" in {
def SVSHRNB      : SInst<"svshrnb[_n_{d}]",    "hdi",  "silUsUiUl", MergeNone, "aarch64_sve_shrnb",     [], [ImmCheck<1, ImmCheckShiftRightNarrow, 0>]>;
}

////////////////////////////////////////////////////////////////////////////////
// SVE2 - Uniform DSP operations
let ArchGuard = "defined(__ARM_FEATURE_SVE2)" in {
def SVQSHLU_M  : SInst<"svqshlu[_n_{d}]", "uPdi", "csil",         MergeOp1,  "aarch64_sve_sqshlu", [], [ImmCheck<2, ImmCheckShiftLeft,  1>]>;
}

////////////////////////////////////////////////////////////////////////////////
// Floating-point arithmetic

defm SVABS_F : SInstZPZ<"svabs", "hfd", "aarch64_sve_fabs">;
defm SVNEG_F : SInstZPZ<"svneg", "hfd", "aarch64_sve_fneg">;

def SVTMAD  : SInst<"svtmad[_{d}]",  "dddi", "hfd", MergeNone, "aarch64_sve_ftmad_x", [], [ImmCheck<2, ImmCheck0_7>]>;

def SVMLA_LANE  : SInst<"svmla_lane[_{d}]",  "ddddi",  "hfd", MergeNone, "aarch64_sve_fmla_lane", [], [ImmCheck<3, ImmCheckLaneIndex, 2>]>;
def SVCMLA_LANE : SInst<"svcmla_lane[_{d}]", "ddddii", "hf",  MergeNone, "aarch64_sve_fcmla_lane", [], [ImmCheck<3, ImmCheckLaneIndexCompRotate, 2>,
                                                                                                        ImmCheck<4, ImmCheckComplexRotAll90>]>;

def SVCADD_M : SInst<"svcadd[_{d}]", "dPddi",  "hfd", MergeOp1,  "aarch64_sve_fcadd", [], [ImmCheck<3, ImmCheckComplexRot90_270>]>;
def SVCMLA_M : SInst<"svcmla[_{d}]", "dPdddi", "hfd", MergeOp1,  "aarch64_sve_fcmla", [], [ImmCheck<4, ImmCheckComplexRotAll90>]>;

////////////////////////////////////////////////////////////////////////////////
// Saturating scalar arithmetic
def SVQDECH_S : SInst<"svqdech_pat[_{d}]",   "ddIi", "s", MergeNone, "aarch64_sve_sqdech", [], [ImmCheck<2, ImmCheck1_16>]>;
def SVQDECH_U : SInst<"svqdech_pat[_{d}]",   "ddIi", "Us", MergeNone, "aarch64_sve_uqdech", [], [ImmCheck<2, ImmCheck1_16>]>;

////////////////////////////////////////////////////////////////////////////////
// Integer arithmetic
def SVDOT_LANE_S : SInst<"svdot_lane[_{d}]",  "ddqqi",  "il",   MergeNone, "aarch64_sve_sdot_lane", [], [ImmCheck<3, ImmCheckLaneIndexDot, 2>]>;
def SVDOT_LANE_U : SInst<"svdot_lane[_{d}]",  "ddqqi",  "UiUl", MergeNone, "aarch64_sve_udot_lane", [], [ImmCheck<3, ImmCheckLaneIndexDot, 2>]>;
