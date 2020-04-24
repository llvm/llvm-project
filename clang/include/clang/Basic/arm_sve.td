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
// v: void
// x: vector of signed integers
// u: vector of unsigned integers
// d: default
// c: const pointer type
// P: predicate type
// a: scalar of element type (splat to vector type)
// e: 1/2 width unsigned elements, 2x element count
// h: 1/2 width elements, 2x element count
// q: 1/4 width elements, 4x element count
// o: 4x width elements, 1/4 element count
//
// i: constant uint64_t
// k: int32_t
// l: int64_t
// m: uint32_t
// n: uint64_t

// t: svint32_t
// z: svuint32_t
// g: svuint64_t
// O: svfloat16_t
// M: svfloat32_t
// N: svfloat64_t

// J: Prefetch type (sv_prfop)
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
def FirstSplatOperand         : FlagType<0x00000400>;
//      :                                     :
// These flags are used to specify which scalar operand
// needs to be duplicated/splatted into a vector.
//      :                                     :
def SplatOperandMask          : FlagType<0x00001C00>;
def IsLoad                    : FlagType<0x00002000>;
def IsStore                   : FlagType<0x00004000>;
def IsGatherLoad              : FlagType<0x00008000>;
def IsScatterStore            : FlagType<0x00010000>;
def IsStructLoad              : FlagType<0x00020000>;
def IsStructStore             : FlagType<0x00040000>;
def IsZExtReturn              : FlagType<0x00080000>; // Return value is sign-extend by default
def IsOverloadNone            : FlagType<0x00100000>; // Intrinsic does not take any overloaded types.
def IsOverloadWhile           : FlagType<0x00200000>; // Use {default type, typeof(operand1)} as overloaded types.
def IsOverloadWhileRW         : FlagType<0x00400000>; // Use {pred(default type), typeof(operand0)} as overloaded types.
def IsOverloadCvt             : FlagType<0x00800000>; // Use {typeof(operand0), typeof(last operand)} as overloaded types.
def OverloadKindMask          : FlagType<0x00E00000>; // When the masked values are all '0', the default type is used as overload type.
def IsByteIndexed             : FlagType<0x01000000>;
def IsPrefetch                : FlagType<0x08000000>; // Contiguous prefetches.

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
def ImmCheck0_13                : ImmCheckType<12>; // 0..13

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
def SVLD1   : MInst<"svld1[_{2}]", "dPc", "csilUcUsUiUlhfd", [IsLoad],               MemEltTyDefault, "aarch64_sve_ld1">;
def SVLD1SB : MInst<"svld1sb_{d}", "dPS", "silUsUiUl",       [IsLoad],               MemEltTyInt8,    "aarch64_sve_ld1">;
def SVLD1UB : MInst<"svld1ub_{d}", "dPW", "silUsUiUl",       [IsLoad, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ld1">;
def SVLD1SH : MInst<"svld1sh_{d}", "dPT", "ilUiUl",          [IsLoad],               MemEltTyInt16,   "aarch64_sve_ld1">;
def SVLD1UH : MInst<"svld1uh_{d}", "dPX", "ilUiUl",          [IsLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1">;
def SVLD1SW : MInst<"svld1sw_{d}", "dPU", "lUl",             [IsLoad],               MemEltTyInt32,   "aarch64_sve_ld1">;
def SVLD1UW : MInst<"svld1uw_{d}", "dPY", "lUl",             [IsLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ld1">;

// Load one vector (scalar base, VL displacement)
def SVLD1_VNUM   : MInst<"svld1_vnum[_{2}]", "dPcl", "csilUcUsUiUlhfd", [IsLoad],               MemEltTyDefault, "aarch64_sve_ld1">;
def SVLD1SB_VNUM : MInst<"svld1sb_vnum_{d}", "dPSl", "silUsUiUl",       [IsLoad],               MemEltTyInt8,    "aarch64_sve_ld1">;
def SVLD1UB_VNUM : MInst<"svld1ub_vnum_{d}", "dPWl", "silUsUiUl",       [IsLoad, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ld1">;
def SVLD1SH_VNUM : MInst<"svld1sh_vnum_{d}", "dPTl", "ilUiUl",          [IsLoad],               MemEltTyInt16,   "aarch64_sve_ld1">;
def SVLD1UH_VNUM : MInst<"svld1uh_vnum_{d}", "dPXl", "ilUiUl",          [IsLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1">;
def SVLD1SW_VNUM : MInst<"svld1sw_vnum_{d}", "dPUl", "lUl",             [IsLoad],               MemEltTyInt32,   "aarch64_sve_ld1">;
def SVLD1UW_VNUM : MInst<"svld1uw_vnum_{d}", "dPYl", "lUl",             [IsLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ld1">;

// Load one vector (vector base)
def SVLD1_GATHER_BASES_U   : MInst<"svld1_gather[_{2}base]_{d}",   "dPu", "ilUiUlfd", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1SB_GATHER_BASES_U : MInst<"svld1sb_gather[_{2}base]_{d}", "dPu", "ilUiUl",   [IsGatherLoad],               MemEltTyInt8,    "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1UB_GATHER_BASES_U : MInst<"svld1ub_gather[_{2}base]_{d}", "dPu", "ilUiUl",   [IsGatherLoad, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1SH_GATHER_BASES_U : MInst<"svld1sh_gather[_{2}base]_{d}", "dPu", "ilUiUl",   [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1UH_GATHER_BASES_U : MInst<"svld1uh_gather[_{2}base]_{d}", "dPu", "ilUiUl",   [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1SW_GATHER_BASES_U : MInst<"svld1sw_gather[_{2}base]_{d}", "dPu", "lUl",      [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1UW_GATHER_BASES_U : MInst<"svld1uw_gather[_{2}base]_{d}", "dPu", "lUl",      [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ld1_gather_scalar_offset">;

// Load one vector (scalar base, signed vector offset in bytes)
def SVLD1_GATHER_64B_OFFSETS_S   : MInst<"svld1_gather_[{3}]offset[_{d}]", "dPcx", "lUld", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ld1_gather">;
def SVLD1SB_GATHER_64B_OFFSETS_S : MInst<"svld1sb_gather_[{3}]offset_{d}", "dPSx", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ld1_gather">;
def SVLD1UB_GATHER_64B_OFFSETS_S : MInst<"svld1ub_gather_[{3}]offset_{d}", "dPWx", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ld1_gather">;
def SVLD1SH_GATHER_64B_OFFSETS_S : MInst<"svld1sh_gather_[{3}]offset_{d}", "dPTx", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ld1_gather">;
def SVLD1UH_GATHER_64B_OFFSETS_S : MInst<"svld1uh_gather_[{3}]offset_{d}", "dPXx", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather">;
def SVLD1SW_GATHER_64B_OFFSETS_S : MInst<"svld1sw_gather_[{3}]offset_{d}", "dPUx", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt32,   "aarch64_sve_ld1_gather">;
def SVLD1UW_GATHER_64B_OFFSETS_S : MInst<"svld1uw_gather_[{3}]offset_{d}", "dPYx", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ld1_gather">;

def SVLD1_GATHER_32B_OFFSETS_S   : MInst<"svld1_gather_[{3}]offset[_{d}]", "dPcx", "iUif", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ld1_gather_sxtw">;
def SVLD1SB_GATHER_32B_OFFSETS_S : MInst<"svld1sb_gather_[{3}]offset_{d}", "dPSx", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ld1_gather_sxtw">;
def SVLD1UB_GATHER_32B_OFFSETS_S : MInst<"svld1ub_gather_[{3}]offset_{d}", "dPWx", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ld1_gather_sxtw">;
def SVLD1SH_GATHER_32B_OFFSETS_S : MInst<"svld1sh_gather_[{3}]offset_{d}", "dPTx", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ld1_gather_sxtw">;
def SVLD1UH_GATHER_32B_OFFSETS_S : MInst<"svld1uh_gather_[{3}]offset_{d}", "dPXx", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather_sxtw">;

// Load one vector (scalar base, unsigned vector offset in bytes)
def SVLD1_GATHER_64B_OFFSETS_U   : MInst<"svld1_gather_[{3}]offset[_{d}]", "dPcu", "lUld", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ld1_gather">;
def SVLD1SB_GATHER_64B_OFFSETS_U : MInst<"svld1sb_gather_[{3}]offset_{d}", "dPSu", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ld1_gather">;
def SVLD1UB_GATHER_64B_OFFSETS_U : MInst<"svld1ub_gather_[{3}]offset_{d}", "dPWu", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ld1_gather">;
def SVLD1SH_GATHER_64B_OFFSETS_U : MInst<"svld1sh_gather_[{3}]offset_{d}", "dPTu", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ld1_gather">;
def SVLD1UH_GATHER_64B_OFFSETS_U : MInst<"svld1uh_gather_[{3}]offset_{d}", "dPXu", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather">;
def SVLD1SW_GATHER_64B_OFFSETS_U : MInst<"svld1sw_gather_[{3}]offset_{d}", "dPUu", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt32,   "aarch64_sve_ld1_gather">;
def SVLD1UW_GATHER_64B_OFFSETS_U : MInst<"svld1uw_gather_[{3}]offset_{d}", "dPYu", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ld1_gather">;

def SVLD1_GATHER_32B_OFFSETS_U   : MInst<"svld1_gather_[{3}]offset[_{d}]", "dPcu", "iUif", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ld1_gather_uxtw">;
def SVLD1SB_GATHER_32B_OFFSETS_U : MInst<"svld1sb_gather_[{3}]offset_{d}", "dPSu", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ld1_gather_uxtw">;
def SVLD1UB_GATHER_32B_OFFSETS_U : MInst<"svld1ub_gather_[{3}]offset_{d}", "dPWu", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ld1_gather_uxtw">;
def SVLD1SH_GATHER_32B_OFFSETS_U : MInst<"svld1sh_gather_[{3}]offset_{d}", "dPTu", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ld1_gather_uxtw">;
def SVLD1UH_GATHER_32B_OFFSETS_U : MInst<"svld1uh_gather_[{3}]offset_{d}", "dPXu", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather_uxtw">;

// Load one vector (vector base, signed scalar offset in bytes)
def SVLD1_GATHER_OFFSET_S   : MInst<"svld1_gather[_{2}base]_offset_{d}",   "dPul", "ilUiUlfd", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1SB_GATHER_OFFSET_S : MInst<"svld1sb_gather[_{2}base]_offset_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1UB_GATHER_OFFSET_S : MInst<"svld1ub_gather[_{2}base]_offset_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1SH_GATHER_OFFSET_S : MInst<"svld1sh_gather[_{2}base]_offset_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1UH_GATHER_OFFSET_S : MInst<"svld1uh_gather[_{2}base]_offset_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1SW_GATHER_OFFSET_S : MInst<"svld1sw_gather[_{2}base]_offset_{d}", "dPul", "lUl",      [IsGatherLoad, IsByteIndexed],               MemEltTyInt32,   "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1UW_GATHER_OFFSET_S : MInst<"svld1uw_gather[_{2}base]_offset_{d}", "dPul", "lUl",      [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ld1_gather_scalar_offset">;

// Load one vector (scalar base, signed vector index)
def SVLD1_GATHER_64B_INDICES_S   : MInst<"svld1_gather_[{3}]index[_{d}]", "dPcx", "lUld", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ld1_gather_index">;
def SVLD1SH_GATHER_64B_INDICES_S : MInst<"svld1sh_gather_[{3}]index_{d}", "dPTx", "lUl",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ld1_gather_index">;
def SVLD1UH_GATHER_64B_INDICES_S : MInst<"svld1uh_gather_[{3}]index_{d}", "dPXx", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather_index">;
def SVLD1SW_GATHER_64B_INDICES_S : MInst<"svld1sw_gather_[{3}]index_{d}", "dPUx", "lUl",  [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ld1_gather_index">;
def SVLD1UW_GATHER_64B_INDICES_S : MInst<"svld1uw_gather_[{3}]index_{d}", "dPYx", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ld1_gather_index">;

def SVLD1_GATHER_32B_INDICES_S   : MInst<"svld1_gather_[{3}]index[_{d}]", "dPcx", "iUif", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ld1_gather_sxtw_index">;
def SVLD1SH_GATHER_32B_INDICES_S : MInst<"svld1sh_gather_[{3}]index_{d}", "dPTx", "iUi",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ld1_gather_sxtw_index">;
def SVLD1UH_GATHER_32B_INDICES_S : MInst<"svld1uh_gather_[{3}]index_{d}", "dPXx", "iUi",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather_sxtw_index">;

// Load one vector (scalar base, unsigned vector index)
def SVLD1_GATHER_64B_INDICES_U   : MInst<"svld1_gather_[{3}]index[_{d}]", "dPcu", "lUld", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ld1_gather_index">;
def SVLD1SH_GATHER_64B_INDICES_U : MInst<"svld1sh_gather_[{3}]index_{d}", "dPTu", "lUl",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ld1_gather_index">;
def SVLD1UH_GATHER_64B_INDICES_U : MInst<"svld1uh_gather_[{3}]index_{d}", "dPXu", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather_index">;
def SVLD1SW_GATHER_64B_INDICES_U : MInst<"svld1sw_gather_[{3}]index_{d}", "dPUu", "lUl",  [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ld1_gather_index">;
def SVLD1UW_GATHER_64B_INDICES_U : MInst<"svld1uw_gather_[{3}]index_{d}", "dPYu", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ld1_gather_index">;

def SVLD1_GATHER_32B_INDICES_U   : MInst<"svld1_gather_[{3}]index[_{d}]", "dPcu", "iUif", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ld1_gather_uxtw_index">;
def SVLD1SH_GATHER_32B_INDICES_U : MInst<"svld1sh_gather_[{3}]index_{d}", "dPTu", "iUi",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ld1_gather_uxtw_index">;
def SVLD1UH_GATHER_32B_INDICES_U : MInst<"svld1uh_gather_[{3}]index_{d}", "dPXu", "iUi",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather_uxtw_index">;

// Load one vector (vector base, signed scalar index)
def SVLD1_GATHER_INDEX_S     : MInst<"svld1_gather[_{2}base]_index_{d}",   "dPul", "ilUiUlfd", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1SH_GATHER_INDEX_S   : MInst<"svld1sh_gather[_{2}base]_index_{d}", "dPul", "ilUiUl",   [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1UH_GATHER_INDEX_S   : MInst<"svld1uh_gather[_{2}base]_index_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1SW_GATHER_INDEX_S   : MInst<"svld1sw_gather[_{2}base]_index_{d}", "dPul", "lUl",      [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ld1_gather_scalar_offset">;
def SVLD1UW_GATHER_INDEX_S   : MInst<"svld1uw_gather[_{2}base]_index_{d}", "dPul", "lUl",      [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ld1_gather_scalar_offset">;


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

// First-faulting load one vector (vector base)
def SVLDFF1_GATHER_BASES_U   : MInst<"svldff1_gather[_{2}base]_{d}",   "dPu", "ilUiUlfd", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1SB_GATHER_BASES_U : MInst<"svldff1sb_gather[_{2}base]_{d}", "dPu", "ilUiUl",   [IsGatherLoad],               MemEltTyInt8,    "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1UB_GATHER_BASES_U : MInst<"svldff1ub_gather[_{2}base]_{d}", "dPu", "ilUiUl",   [IsGatherLoad, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1SH_GATHER_BASES_U : MInst<"svldff1sh_gather[_{2}base]_{d}", "dPu", "ilUiUl",   [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1UH_GATHER_BASES_U : MInst<"svldff1uh_gather[_{2}base]_{d}", "dPu", "ilUiUl",   [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1SW_GATHER_BASES_U : MInst<"svldff1sw_gather[_{2}base]_{d}", "dPu", "lUl",      [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1UW_GATHER_BASES_U : MInst<"svldff1uw_gather[_{2}base]_{d}", "dPu", "lUl",      [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldff1_gather_scalar_offset">;

// First-faulting load one vector (scalar base, signed vector offset in bytes)
def SVLDFF1_GATHER_64B_OFFSETS_S   : MInst<"svldff1_gather_[{3}]offset[_{d}]", "dPcx", "lUld", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ldff1_gather">;
def SVLDFF1SB_GATHER_64B_OFFSETS_S : MInst<"svldff1sb_gather_[{3}]offset_{d}", "dPSx", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ldff1_gather">;
def SVLDFF1UB_GATHER_64B_OFFSETS_S : MInst<"svldff1ub_gather_[{3}]offset_{d}", "dPWx", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldff1_gather">;
def SVLDFF1SH_GATHER_64B_OFFSETS_S : MInst<"svldff1sh_gather_[{3}]offset_{d}", "dPTx", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ldff1_gather">;
def SVLDFF1UH_GATHER_64B_OFFSETS_S : MInst<"svldff1uh_gather_[{3}]offset_{d}", "dPXx", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather">;
def SVLDFF1SW_GATHER_64B_OFFSETS_S : MInst<"svldff1sw_gather_[{3}]offset_{d}", "dPUx", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt32,   "aarch64_sve_ldff1_gather">;
def SVLDFF1UW_GATHER_64B_OFFSETS_S : MInst<"svldff1uw_gather_[{3}]offset_{d}", "dPYx", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldff1_gather">;

def SVLDFF1_GATHER_32B_OFFSETS_S   : MInst<"svldff1_gather_[{3}]offset[_{d}]", "dPcx", "iUif", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ldff1_gather_sxtw">;
def SVLDFF1SB_GATHER_32B_OFFSETS_S : MInst<"svldff1sb_gather_[{3}]offset_{d}", "dPSx", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ldff1_gather_sxtw">;
def SVLDFF1UB_GATHER_32B_OFFSETS_S : MInst<"svldff1ub_gather_[{3}]offset_{d}", "dPWx", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldff1_gather_sxtw">;
def SVLDFF1SH_GATHER_32B_OFFSETS_S : MInst<"svldff1sh_gather_[{3}]offset_{d}", "dPTx", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ldff1_gather_sxtw">;
def SVLDFF1UH_GATHER_32B_OFFSETS_S : MInst<"svldff1uh_gather_[{3}]offset_{d}", "dPXx", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather_sxtw">;

// First-faulting load one vector (scalar base, unsigned vector offset in bytes)
def SVLDFF1_GATHER_64B_OFFSETS_U   : MInst<"svldff1_gather_[{3}]offset[_{d}]", "dPcu", "lUld", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ldff1_gather">;
def SVLDFF1SB_GATHER_64B_OFFSETS_U : MInst<"svldff1sb_gather_[{3}]offset_{d}", "dPSu", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ldff1_gather">;
def SVLDFF1UB_GATHER_64B_OFFSETS_U : MInst<"svldff1ub_gather_[{3}]offset_{d}", "dPWu", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldff1_gather">;
def SVLDFF1SH_GATHER_64B_OFFSETS_U : MInst<"svldff1sh_gather_[{3}]offset_{d}", "dPTu", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ldff1_gather">;
def SVLDFF1UH_GATHER_64B_OFFSETS_U : MInst<"svldff1uh_gather_[{3}]offset_{d}", "dPXu", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather">;
def SVLDFF1SW_GATHER_64B_OFFSETS_U : MInst<"svldff1sw_gather_[{3}]offset_{d}", "dPUu", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt32,   "aarch64_sve_ldff1_gather">;
def SVLDFF1UW_GATHER_64B_OFFSETS_U : MInst<"svldff1uw_gather_[{3}]offset_{d}", "dPYu", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldff1_gather">;

def SVLDFF1_GATHER_32B_OFFSETS_U   : MInst<"svldff1_gather_[{3}]offset[_{d}]", "dPcu", "iUif", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ldff1_gather_uxtw">;
def SVLDFF1SB_GATHER_32B_OFFSETS_U : MInst<"svldff1sb_gather_[{3}]offset_{d}", "dPSu", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ldff1_gather_uxtw">;
def SVLDFF1UB_GATHER_32B_OFFSETS_U : MInst<"svldff1ub_gather_[{3}]offset_{d}", "dPWu", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldff1_gather_uxtw">;
def SVLDFF1SH_GATHER_32B_OFFSETS_U : MInst<"svldff1sh_gather_[{3}]offset_{d}", "dPTu", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ldff1_gather_uxtw">;
def SVLDFF1UH_GATHER_32B_OFFSETS_U : MInst<"svldff1uh_gather_[{3}]offset_{d}", "dPXu", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather_uxtw">;

// First-faulting load one vector (vector base, signed scalar offset in bytes)
def SVLDFF1_GATHER_OFFSET_S   : MInst<"svldff1_gather[_{2}base]_offset_{d}",   "dPul", "ilUiUlfd", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1SB_GATHER_OFFSET_S : MInst<"svldff1sb_gather[_{2}base]_offset_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1UB_GATHER_OFFSET_S : MInst<"svldff1ub_gather[_{2}base]_offset_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1SH_GATHER_OFFSET_S : MInst<"svldff1sh_gather[_{2}base]_offset_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1UH_GATHER_OFFSET_S : MInst<"svldff1uh_gather[_{2}base]_offset_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1SW_GATHER_OFFSET_S : MInst<"svldff1sw_gather[_{2}base]_offset_{d}", "dPul", "lUl",      [IsGatherLoad, IsByteIndexed],               MemEltTyInt32,   "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1UW_GATHER_OFFSET_S : MInst<"svldff1uw_gather[_{2}base]_offset_{d}", "dPul", "lUl",      [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldff1_gather_scalar_offset">;

// First-faulting load one vector (scalar base, signed vector index)
def SVLDFF1_GATHER_64B_INDICES_S   : MInst<"svldff1_gather_[{3}]index[_{d}]", "dPcx", "lUld", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldff1_gather_index">;
def SVLDFF1SH_GATHER_64B_INDICES_S : MInst<"svldff1sh_gather_[{3}]index_{d}", "dPTx", "lUl",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldff1_gather_index">;
def SVLDFF1UH_GATHER_64B_INDICES_S : MInst<"svldff1uh_gather_[{3}]index_{d}", "dPXx", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather_index">;
def SVLDFF1SW_GATHER_64B_INDICES_S : MInst<"svldff1sw_gather_[{3}]index_{d}", "dPUx", "lUl",  [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ldff1_gather_index">;
def SVLDFF1UW_GATHER_64B_INDICES_S : MInst<"svldff1uw_gather_[{3}]index_{d}", "dPYx", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldff1_gather_index">;

def SVLDFF1_GATHER_32B_INDICES_S   : MInst<"svldff1_gather_[{3}]index[_{d}]", "dPcx", "iUif", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldff1_gather_sxtw_index">;
def SVLDFF1SH_GATHER_32B_INDICES_S : MInst<"svldff1sh_gather_[{3}]index_{d}", "dPTx", "iUi",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldff1_gather_sxtw_index">;
def SVLDFF1UH_GATHER_32B_INDICES_S : MInst<"svldff1uh_gather_[{3}]index_{d}", "dPXx", "iUi",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather_sxtw_index">;

// First-faulting load one vector (scalar base, unsigned vector index)
def SVLDFF1_GATHER_64B_INDICES_U   : MInst<"svldff1_gather_[{3}]index[_{d}]", "dPcu", "lUld", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldff1_gather_index">;
def SVLDFF1SH_GATHER_64B_INDICES_U : MInst<"svldff1sh_gather_[{3}]index_{d}", "dPTu", "lUl",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldff1_gather_index">;
def SVLDFF1UH_GATHER_64B_INDICES_U : MInst<"svldff1uh_gather_[{3}]index_{d}", "dPXu", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather_index">;
def SVLDFF1SW_GATHER_64B_INDICES_U : MInst<"svldff1sw_gather_[{3}]index_{d}", "dPUu", "lUl",  [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ldff1_gather_index">;
def SVLDFF1UW_GATHER_64B_INDICES_U : MInst<"svldff1uw_gather_[{3}]index_{d}", "dPYu", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldff1_gather_index">;

def SVLDFF1_GATHER_32B_INDICES_U   : MInst<"svldff1_gather_[{3}]index[_{d}]", "dPcu", "iUif", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldff1_gather_uxtw_index">;
def SVLDFF1SH_GATHER_32B_INDICES_U : MInst<"svldff1sh_gather_[{3}]index_{d}", "dPTu", "iUi",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldff1_gather_uxtw_index">;
def SVLDFF1UH_GATHER_32B_INDICES_U : MInst<"svldff1uh_gather_[{3}]index_{d}", "dPXu", "iUi",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather_uxtw_index">;

// First-faulting load one vector (vector base, signed scalar index)
def SVLDFF1_GATHER_INDEX_S   : MInst<"svldff1_gather[_{2}base]_index_{d}",   "dPul", "ilUiUlfd", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1SH_GATHER_INDEX_S : MInst<"svldff1sh_gather[_{2}base]_index_{d}", "dPul", "ilUiUl",   [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1UH_GATHER_INDEX_S : MInst<"svldff1uh_gather[_{2}base]_index_{d}", "dPul", "ilUiUl",   [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1SW_GATHER_INDEX_S : MInst<"svldff1sw_gather[_{2}base]_index_{d}", "dPul", "lUl",      [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ldff1_gather_scalar_offset">;
def SVLDFF1UW_GATHER_INDEX_S : MInst<"svldff1uw_gather[_{2}base]_index_{d}", "dPul", "lUl",      [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldff1_gather_scalar_offset">;

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

// Load one quadword and replicate (scalar base)
def SVLD1RQ : SInst<"svld1rq[_{2}]", "dPc", "csilUcUsUiUlhfd", MergeNone, "aarch64_sve_ld1rq">;

////////////////////////////////////////////////////////////////////////////////
// Stores

// Store one vector (scalar base)
def SVST1    : MInst<"svst1[_{d}]",  "vPpd", "csilUcUsUiUlhfd", [IsStore], MemEltTyDefault, "aarch64_sve_st1">;
def SVST1B_S : MInst<"svst1b[_{d}]", "vPAd", "sil",             [IsStore], MemEltTyInt8,    "aarch64_sve_st1">;
def SVST1B_U : MInst<"svst1b[_{d}]", "vPEd", "UsUiUl",          [IsStore], MemEltTyInt8,    "aarch64_sve_st1">;
def SVST1H_S : MInst<"svst1h[_{d}]", "vPBd", "il",              [IsStore], MemEltTyInt16,   "aarch64_sve_st1">;
def SVST1H_U : MInst<"svst1h[_{d}]", "vPFd", "UiUl",            [IsStore], MemEltTyInt16,   "aarch64_sve_st1">;
def SVST1W_S : MInst<"svst1w[_{d}]", "vPCd", "l",               [IsStore], MemEltTyInt32,   "aarch64_sve_st1">;
def SVST1W_U : MInst<"svst1w[_{d}]", "vPGd", "Ul",              [IsStore], MemEltTyInt32,   "aarch64_sve_st1">;

// Store one vector (scalar base, VL displacement)
def SVST1_VNUM    : MInst<"svst1_vnum[_{d}]",  "vPpld", "csilUcUsUiUlhfd", [IsStore], MemEltTyDefault, "aarch64_sve_st1">;
def SVST1B_VNUM_S : MInst<"svst1b_vnum[_{d}]", "vPAld", "sil",             [IsStore], MemEltTyInt8,    "aarch64_sve_st1">;
def SVST1B_VNUM_U : MInst<"svst1b_vnum[_{d}]", "vPEld", "UsUiUl",          [IsStore], MemEltTyInt8,    "aarch64_sve_st1">;
def SVST1H_VNUM_S : MInst<"svst1h_vnum[_{d}]", "vPBld", "il",              [IsStore], MemEltTyInt16,   "aarch64_sve_st1">;
def SVST1H_VNUM_U : MInst<"svst1h_vnum[_{d}]", "vPFld", "UiUl",            [IsStore], MemEltTyInt16,   "aarch64_sve_st1">;
def SVST1W_VNUM_S : MInst<"svst1w_vnum[_{d}]", "vPCld", "l",               [IsStore], MemEltTyInt32,   "aarch64_sve_st1">;
def SVST1W_VNUM_U : MInst<"svst1w_vnum[_{d}]", "vPGld", "Ul",              [IsStore], MemEltTyInt32,   "aarch64_sve_st1">;

// Store one vector (vector base)
def SVST1_SCATTER_BASES_U     : MInst<"svst1_scatter[_{2}base_{d}]",  "vPud",  "ilUiUlfd", [IsScatterStore], MemEltTyDefault, "aarch64_sve_st1_scatter_scalar_offset">;
def SVST1B_SCATTER_BASES_U    : MInst<"svst1b_scatter[_{2}base_{d}]", "vPud",  "ilUiUl",   [IsScatterStore], MemEltTyInt8,    "aarch64_sve_st1_scatter_scalar_offset">;
def SVST1H_SCATTER_BASES_U    : MInst<"svst1h_scatter[_{2}base_{d}]", "vPud",  "ilUiUl",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_scalar_offset">;
def SVST1W_SCATTER_BASES_U    : MInst<"svst1w_scatter[_{2}base_{d}]", "vPud",  "lUl",      [IsScatterStore], MemEltTyInt32,   "aarch64_sve_st1_scatter_scalar_offset">;

// Store one vector (scalar base, signed vector offset in bytes)
def SVST1_SCATTER_64B_OFFSETS_S   : MInst<"svst1_scatter_[{3}]offset[_{d}]",  "vPpxd", "lUld", [IsScatterStore, IsByteIndexed], MemEltTyDefault, "aarch64_sve_st1_scatter">;
def SVST1B_SCATTER_64B_OFFSETS_SS : MInst<"svst1b_scatter_[{3}]offset[_{d}]", "vPAxd", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_st1_scatter">;
def SVST1B_SCATTER_64B_OFFSETS_SU : MInst<"svst1b_scatter_[{3}]offset[_{d}]", "vPExd", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_st1_scatter">;
def SVST1H_SCATTER_64B_OFFSETS_SS : MInst<"svst1h_scatter_[{3}]offset[_{d}]", "vPBxd", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_st1_scatter">;
def SVST1H_SCATTER_64B_OFFSETS_SU : MInst<"svst1h_scatter_[{3}]offset[_{d}]", "vPFxd", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_st1_scatter">;
def SVST1W_SCATTER_64B_OFFSETS_SS : MInst<"svst1w_scatter_[{3}]offset[_{d}]", "vPCxd", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_st1_scatter">;
def SVST1W_SCATTER_64B_OFFSETS_SU : MInst<"svst1w_scatter_[{3}]offset[_{d}]", "vPGxd", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_st1_scatter">;

def SVST1_SCATTER_32B_OFFSETS_S   : MInst<"svst1_scatter_[{3}]offset[_{d}]",  "vPpxd", "iUif", [IsScatterStore, IsByteIndexed], MemEltTyDefault, "aarch64_sve_st1_scatter_sxtw">;
def SVST1B_SCATTER_32B_OFFSETS_SS : MInst<"svst1b_scatter_[{3}]offset[_{d}]", "vPAxd", "i",    [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_st1_scatter_sxtw">;
def SVST1B_SCATTER_32B_OFFSETS_SU : MInst<"svst1b_scatter_[{3}]offset[_{d}]", "vPExd", "Ui",   [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_st1_scatter_sxtw">;
def SVST1H_SCATTER_32B_OFFSETS_SS : MInst<"svst1h_scatter_[{3}]offset[_{d}]", "vPBxd", "i",    [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_st1_scatter_sxtw">;
def SVST1H_SCATTER_32B_OFFSETS_SU : MInst<"svst1h_scatter_[{3}]offset[_{d}]", "vPFxd", "Ui",   [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_st1_scatter_sxtw">;

// Store one vector (scalar base, unsigned vector offset in bytes)
def SVST1_SCATTER_64B_OFFSETS_U   : MInst<"svst1_scatter_[{3}]offset[_{d}]",  "vPpud", "lUld", [IsScatterStore, IsByteIndexed], MemEltTyDefault, "aarch64_sve_st1_scatter">;
def SVST1B_SCATTER_64B_OFFSETS_US : MInst<"svst1b_scatter_[{3}]offset[_{d}]", "vPAud", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_st1_scatter">;
def SVST1B_SCATTER_64B_OFFSETS_UU : MInst<"svst1b_scatter_[{3}]offset[_{d}]", "vPEud", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_st1_scatter">;
def SVST1H_SCATTER_64B_OFFSETS_US : MInst<"svst1h_scatter_[{3}]offset[_{d}]", "vPBud", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_st1_scatter">;
def SVST1H_SCATTER_64B_OFFSETS_UU : MInst<"svst1h_scatter_[{3}]offset[_{d}]", "vPFud", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_st1_scatter">;
def SVST1W_SCATTER_64B_OFFSETS_US : MInst<"svst1w_scatter_[{3}]offset[_{d}]", "vPCud", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_st1_scatter">;
def SVST1W_SCATTER_64B_OFFSETS_UU : MInst<"svst1w_scatter_[{3}]offset[_{d}]", "vPGud", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_st1_scatter">;

def SVST1_SCATTER_32B_OFFSETS_U   : MInst<"svst1_scatter_[{3}]offset[_{d}]",  "vPpud", "iUif", [IsScatterStore, IsByteIndexed], MemEltTyDefault, "aarch64_sve_st1_scatter_uxtw">;
def SVST1B_SCATTER_32B_OFFSETS_US : MInst<"svst1b_scatter_[{3}]offset[_{d}]", "vPAud", "i",    [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_st1_scatter_uxtw">;
def SVST1B_SCATTER_32B_OFFSETS_UU : MInst<"svst1b_scatter_[{3}]offset[_{d}]", "vPEud", "Ui",   [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_st1_scatter_uxtw">;
def SVST1H_SCATTER_32B_OFFSETS_US : MInst<"svst1h_scatter_[{3}]offset[_{d}]", "vPBud", "i",    [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_st1_scatter_uxtw">;
def SVST1H_SCATTER_32B_OFFSETS_UU : MInst<"svst1h_scatter_[{3}]offset[_{d}]", "vPFud", "Ui",   [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_st1_scatter_uxtw">;

// Store one vector (vector base, signed scalar offset in bytes)
def SVST1_SCATTER_OFFSET_S    : MInst<"svst1_scatter[_{2}base]_offset[_{d}]",  "vPuld", "ilUiUlfd", [IsScatterStore, IsByteIndexed], MemEltTyDefault, "aarch64_sve_st1_scatter_scalar_offset">;
def SVST1B_SCATTER_OFFSET_S   : MInst<"svst1b_scatter[_{2}base]_offset[_{d}]", "vPuld", "ilUiUl",   [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_st1_scatter_scalar_offset">;
def SVST1H_SCATTER_OFFSET_S   : MInst<"svst1h_scatter[_{2}base]_offset[_{d}]", "vPuld", "ilUiUl",   [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_st1_scatter_scalar_offset">;
def SVST1W_SCATTER_OFFSET_S   : MInst<"svst1w_scatter[_{2}base]_offset[_{d}]", "vPuld", "lUl",      [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_st1_scatter_scalar_offset">;

// Store one vector (scalar base, signed vector index)
def SVST1_SCATTER_64B_INDICES_S   : MInst<"svst1_scatter_[{3}]index[_{d}]",  "vPpxd", "lUld", [IsScatterStore], MemEltTyDefault, "aarch64_sve_st1_scatter_index">;
def SVST1H_SCATTER_64B_INDICES_SS : MInst<"svst1h_scatter_[{3}]index[_{d}]", "vPBxd", "l",    [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_index">;
def SVST1H_SCATTER_64B_INDICES_SU : MInst<"svst1h_scatter_[{3}]index[_{d}]", "vPFxd", "Ul",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_index">;
def SVST1W_SCATTER_64B_INDICES_SS : MInst<"svst1w_scatter_[{3}]index[_{d}]", "vPCxd", "l",    [IsScatterStore], MemEltTyInt32,   "aarch64_sve_st1_scatter_index">;
def SVST1W_SCATTER_64B_INDICES_SU : MInst<"svst1w_scatter_[{3}]index[_{d}]", "vPGxd", "Ul",   [IsScatterStore], MemEltTyInt32,   "aarch64_sve_st1_scatter_index">;

def SVST1_SCATTER_32B_INDICES_S   : MInst<"svst1_scatter_[{3}]index[_{d}]",  "vPpxd", "iUif", [IsScatterStore], MemEltTyDefault, "aarch64_sve_st1_scatter_sxtw_index">;
def SVST1H_SCATTER_32B_INDICES_SS : MInst<"svst1h_scatter_[{3}]index[_{d}]", "vPBxd", "i",    [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_sxtw_index">;
def SVST1H_SCATTER_32B_INDICES_SU : MInst<"svst1h_scatter_[{3}]index[_{d}]", "vPFxd", "Ui",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_sxtw_index">;

// Store one vector (scalar base, unsigned vector index)
def SVST1_SCATTER_64B_INDICES_U   : MInst<"svst1_scatter_[{3}]index[_{d}]",  "vPpud", "lUld", [IsScatterStore], MemEltTyDefault, "aarch64_sve_st1_scatter_index">;
def SVST1H_SCATTER_64B_INDICES_US : MInst<"svst1h_scatter_[{3}]index[_{d}]", "vPBud", "l",    [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_index">;
def SVST1H_SCATTER_64B_INDICES_UU : MInst<"svst1h_scatter_[{3}]index[_{d}]", "vPFud", "Ul",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_index">;
def SVST1W_SCATTER_64B_INDICES_US : MInst<"svst1w_scatter_[{3}]index[_{d}]", "vPCud", "l",    [IsScatterStore], MemEltTyInt32,   "aarch64_sve_st1_scatter_index">;
def SVST1W_SCATTER_64B_INDICES_UU : MInst<"svst1w_scatter_[{3}]index[_{d}]", "vPGud", "Ul",   [IsScatterStore], MemEltTyInt32,   "aarch64_sve_st1_scatter_index">;

def SVST1_SCATTER_32B_INDICES_U   : MInst<"svst1_scatter_[{3}]index[_{d}]",  "vPpud", "iUif", [IsScatterStore], MemEltTyDefault, "aarch64_sve_st1_scatter_uxtw_index">;
def SVST1H_SCATTER_32B_INDICES_US : MInst<"svst1h_scatter_[{3}]index[_{d}]", "vPBud", "i",    [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_uxtw_index">;
def SVST1H_SCATTER_32B_INDICES_UU : MInst<"svst1h_scatter_[{3}]index[_{d}]", "vPFud", "Ui",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_uxtw_index">;

// Store one vector (vector base, signed scalar index)
def SVST1_SCATTER_INDEX_S     : MInst<"svst1_scatter[_{2}base]_index[_{d}]",  "vPuld", "ilUiUlfd", [IsScatterStore], MemEltTyDefault, "aarch64_sve_st1_scatter_scalar_offset">;
def SVST1H_SCATTER_INDEX_S    : MInst<"svst1h_scatter[_{2}base]_index[_{d}]", "vPuld", "ilUiUl",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_st1_scatter_scalar_offset">;
def SVST1W_SCATTER_INDEX_S    : MInst<"svst1w_scatter[_{2}base]_index[_{d}]", "vPuld", "lUl",      [IsScatterStore], MemEltTyInt32,   "aarch64_sve_st1_scatter_scalar_offset">;

// Store one vector, with no truncation, non-temporal (scalar base)
def SVSTNT1 : MInst<"svstnt1[_{d}]", "vPpd", "csilUcUsUiUlhfd", [IsStore], MemEltTyDefault, "aarch64_sve_stnt1">;

// Store one vector, with no truncation, non-temporal (scalar base, VL displacement)
def SVSTNT1_VNUM : MInst<"svstnt1_vnum[_{d}]", "vPpld", "csilUcUsUiUlhfd", [IsStore], MemEltTyDefault, "aarch64_sve_stnt1">;

////////////////////////////////////////////////////////////////////////////////
// Prefetches

// Prefetch (Scalar base)
def SVPRFB : MInst<"svprfb", "vPcJ", "c", [IsPrefetch], MemEltTyInt8,  "aarch64_sve_prf">;
def SVPRFH : MInst<"svprfh", "vPcJ", "s", [IsPrefetch], MemEltTyInt16, "aarch64_sve_prf">;
def SVPRFW : MInst<"svprfw", "vPcJ", "i", [IsPrefetch], MemEltTyInt32, "aarch64_sve_prf">;
def SVPRFD : MInst<"svprfd", "vPcJ", "l", [IsPrefetch], MemEltTyInt64, "aarch64_sve_prf">;

// Prefetch (Scalar base, VL displacement)
def SVPRFB_VNUM : MInst<"svprfb_vnum", "vPclJ", "c", [IsPrefetch], MemEltTyInt8,  "aarch64_sve_prf">;
def SVPRFH_VNUM : MInst<"svprfh_vnum", "vPclJ", "s", [IsPrefetch], MemEltTyInt16, "aarch64_sve_prf">;
def SVPRFW_VNUM : MInst<"svprfw_vnum", "vPclJ", "i", [IsPrefetch], MemEltTyInt32, "aarch64_sve_prf">;
def SVPRFD_VNUM : MInst<"svprfd_vnum", "vPclJ", "l", [IsPrefetch], MemEltTyInt64, "aarch64_sve_prf">;

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

  def _N_M : SInst<name # "[_n_{d}]", "dPda", types, MergeOp1,  intrinsic, flags>;
  def _N_X : SInst<name # "[_n_{d}]", "dPda", types, MergeAny,  intrinsic, flags>;
  def _N_Z : SInst<name # "[_n_{d}]", "dPda", types, MergeZero, intrinsic, flags>;
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
// While comparisons

def SVWHILELE_S32 : SInst<"svwhilele_{d}[_{1}]", "Pkk", "PcPsPiPl",     MergeNone, "aarch64_sve_whilele", [IsOverloadWhile]>;
def SVWHILELE_S64 : SInst<"svwhilele_{d}[_{1}]", "Pll", "PcPsPiPl",     MergeNone, "aarch64_sve_whilele", [IsOverloadWhile]>;
def SVWHILELO_U32 : SInst<"svwhilelt_{d}[_{1}]", "Pmm", "PUcPUsPUiPUl", MergeNone, "aarch64_sve_whilelo", [IsOverloadWhile]>;
def SVWHILELO_U64 : SInst<"svwhilelt_{d}[_{1}]", "Pnn", "PUcPUsPUiPUl", MergeNone, "aarch64_sve_whilelo", [IsOverloadWhile]>;
def SVWHILELS_U32 : SInst<"svwhilele_{d}[_{1}]", "Pmm", "PUcPUsPUiPUl", MergeNone, "aarch64_sve_whilels", [IsOverloadWhile]>;
def SVWHILELS_U64 : SInst<"svwhilele_{d}[_{1}]", "Pnn", "PUcPUsPUiPUl", MergeNone, "aarch64_sve_whilels", [IsOverloadWhile]>;
def SVWHILELT_S32 : SInst<"svwhilelt_{d}[_{1}]", "Pkk", "PcPsPiPl",     MergeNone, "aarch64_sve_whilelt", [IsOverloadWhile]>;
def SVWHILELT_S64 : SInst<"svwhilelt_{d}[_{1}]", "Pll", "PcPsPiPl",     MergeNone, "aarch64_sve_whilelt", [IsOverloadWhile]>;

////////////////////////////////////////////////////////////////////////////////
// Floating-point arithmetic

defm SVABS_F : SInstZPZ<"svabs", "hfd", "aarch64_sve_fabs">;
defm SVNEG_F : SInstZPZ<"svneg", "hfd", "aarch64_sve_fneg">;

def SVTMAD  : SInst<"svtmad[_{d}]",  "dddi", "hfd", MergeNone, "aarch64_sve_ftmad_x", [], [ImmCheck<2, ImmCheck0_7>]>;

def SVMLA_LANE  : SInst<"svmla_lane[_{d}]",  "ddddi",  "hfd", MergeNone, "aarch64_sve_fmla_lane", [], [ImmCheck<3, ImmCheckLaneIndex, 2>]>;
def SVCMLA_LANE : SInst<"svcmla_lane[_{d}]", "ddddii", "hf",  MergeNone, "aarch64_sve_fcmla_lane", [], [ImmCheck<3, ImmCheckLaneIndexCompRotate, 2>,
                                                                                                        ImmCheck<4, ImmCheckComplexRotAll90>]>;
////////////////////////////////////////////////////////////////////////////////
// Floating-point conversions

multiclass SInstCvtMXZ<
    string name, string m_types, string xz_types, string types,
    string intrinsic, list<FlagType> flags = [IsOverloadNone]> {
  def _M : SInst<name, m_types,  types, MergeOp1,     intrinsic, flags>;
  def _X : SInst<name, xz_types, types, MergeAnyExp,  intrinsic, flags>;
  def _Z : SInst<name, xz_types, types, MergeZeroExp, intrinsic, flags>;
}

multiclass SInstCvtMX<string name, string m_types, string xz_types,
                      string types, string intrinsic,
                      list<FlagType> flags = [IsOverloadNone]> {
  def _M : SInst<name, m_types,  types, MergeOp1,     intrinsic, flags>;
  def _X : SInst<name, xz_types, types, MergeAnyExp,  intrinsic, flags>;
}

// svcvt_s##_f16
defm SVFCVTZS_S16_F16 : SInstCvtMXZ<"svcvt_s16[_f16]", "ddPO", "dPO", "s",  "aarch64_sve_fcvtzs", [IsOverloadCvt]>;
defm SVFCVTZS_S32_F16 : SInstCvtMXZ<"svcvt_s32[_f16]", "ddPO", "dPO", "i",  "aarch64_sve_fcvtzs_i32f16">;
defm SVFCVTZS_S64_F16 : SInstCvtMXZ<"svcvt_s64[_f16]", "ddPO", "dPO", "l",  "aarch64_sve_fcvtzs_i64f16">;

// svcvt_s##_f32
defm SVFCVTZS_S32_F32 : SInstCvtMXZ<"svcvt_s32[_f32]", "ddPM", "dPM", "i",  "aarch64_sve_fcvtzs", [IsOverloadCvt]>;
defm SVFCVTZS_S64_F32 : SInstCvtMXZ<"svcvt_s64[_f32]", "ddPM", "dPM", "l",  "aarch64_sve_fcvtzs_i64f32">;

// svcvt_s##_f64
defm SVFCVTZS_S32_F64 : SInstCvtMXZ<"svcvt_s32[_f64]", "ttPd", "tPd", "d",  "aarch64_sve_fcvtzs_i32f64">;
defm SVFCVTZS_S64_F64 : SInstCvtMXZ<"svcvt_s64[_f64]", "ddPN", "dPN", "l",  "aarch64_sve_fcvtzs", [IsOverloadCvt]>;

// svcvt_u##_f16
defm SVFCVTZU_U16_F16 : SInstCvtMXZ<"svcvt_u16[_f16]", "ddPO", "dPO", "Us", "aarch64_sve_fcvtzu", [IsOverloadCvt]>;
defm SVFCVTZU_U32_F16 : SInstCvtMXZ<"svcvt_u32[_f16]", "ddPO", "dPO", "Ui", "aarch64_sve_fcvtzu_i32f16">;
defm SVFCVTZU_U64_F16 : SInstCvtMXZ<"svcvt_u64[_f16]", "ddPO", "dPO", "Ul", "aarch64_sve_fcvtzu_i64f16">;

// svcvt_u##_f32
defm SVFCVTZU_U32_F32 : SInstCvtMXZ<"svcvt_u32[_f32]", "ddPM", "dPM", "Ui", "aarch64_sve_fcvtzu", [IsOverloadCvt]>;
defm SVFCVTZU_U64_F32 : SInstCvtMXZ<"svcvt_u64[_f32]", "ddPM", "dPM", "Ul", "aarch64_sve_fcvtzu_i64f32">;

// svcvt_u##_f64
defm SVFCVTZU_U32_F64 : SInstCvtMXZ<"svcvt_u32[_f64]", "zzPd", "zPd", "d",  "aarch64_sve_fcvtzu_i32f64">;
defm SVFCVTZU_U64_F64 : SInstCvtMXZ<"svcvt_u64[_f64]", "ddPN", "dPN", "Ul", "aarch64_sve_fcvtzu", [IsOverloadCvt]>;

// svcvt_f16_s##
defm SVFCVTZS_F16_S16 : SInstCvtMXZ<"svcvt_f16[_s16]", "OOPd", "OPd", "s",  "aarch64_sve_scvtf", [IsOverloadCvt]>;
defm SVFCVTZS_F16_S32 : SInstCvtMXZ<"svcvt_f16[_s32]", "OOPd", "OPd", "i",  "aarch64_sve_scvtf_f16i32">;
defm SVFCVTZS_F16_S64 : SInstCvtMXZ<"svcvt_f16[_s64]", "OOPd", "OPd", "l",  "aarch64_sve_scvtf_f16i64">;

// svcvt_f32_s##
defm SVFCVTZS_F32_S32 : SInstCvtMXZ<"svcvt_f32[_s32]", "MMPd", "MPd", "i",  "aarch64_sve_scvtf", [IsOverloadCvt]>;
defm SVFCVTZS_F32_S64 : SInstCvtMXZ<"svcvt_f32[_s64]", "MMPd", "MPd", "l",  "aarch64_sve_scvtf_f32i64">;

// svcvt_f64_s##
defm SVFCVTZS_F64_S32 : SInstCvtMXZ<"svcvt_f64[_s32]", "ddPt", "dPt", "d",  "aarch64_sve_scvtf_f64i32">;
defm SVFCVTZS_F64_S64 : SInstCvtMXZ<"svcvt_f64[_s64]", "NNPd", "NPd", "l",  "aarch64_sve_scvtf", [IsOverloadCvt]>;

// svcvt_f16_u##
defm SVFCVTZU_F16_U16 : SInstCvtMXZ<"svcvt_f16[_u16]", "OOPd", "OPd", "Us", "aarch64_sve_ucvtf", [IsOverloadCvt]>;
defm SVFCVTZU_F16_U32 : SInstCvtMXZ<"svcvt_f16[_u32]", "OOPd", "OPd", "Ui", "aarch64_sve_ucvtf_f16i32">;
defm SVFCVTZU_F16_U64 : SInstCvtMXZ<"svcvt_f16[_u64]", "OOPd", "OPd", "Ul", "aarch64_sve_ucvtf_f16i64">;

// svcvt_f32_u##
defm SVFCVTZU_F32_U32 : SInstCvtMXZ<"svcvt_f32[_u32]", "MMPd", "MPd", "Ui", "aarch64_sve_ucvtf", [IsOverloadCvt]>;
defm SVFCVTZU_F32_U64 : SInstCvtMXZ<"svcvt_f32[_u64]", "MMPd", "MPd", "Ul", "aarch64_sve_ucvtf_f32i64">;

// svcvt_f64_u##
defm SVFCVTZU_F64_U32 : SInstCvtMXZ<"svcvt_f64[_u32]", "ddPz", "dPz", "d",  "aarch64_sve_ucvtf_f64i32">;
defm SVFCVTZU_F64_U64 : SInstCvtMXZ<"svcvt_f64[_u64]", "NNPd", "NPd", "Ul", "aarch64_sve_ucvtf", [IsOverloadCvt]>;

// svcvt_f16_f##
defm SVFCVT_F16_F32   : SInstCvtMXZ<"svcvt_f16[_f32]", "OOPd", "OPd", "f", "aarch64_sve_fcvt_f16f32">;
defm SVFCVT_F16_F64   : SInstCvtMXZ<"svcvt_f16[_f64]", "OOPd", "OPd", "d", "aarch64_sve_fcvt_f16f64">;

// svcvt_f32_f##
defm SVFCVT_F32_F16   : SInstCvtMXZ<"svcvt_f32[_f16]", "ddPO", "dPO", "f", "aarch64_sve_fcvt_f32f16">;
defm SVFCVT_F32_F64   : SInstCvtMXZ<"svcvt_f32[_f64]", "MMPd", "MPd", "d", "aarch64_sve_fcvt_f32f64">;

// svcvt_f64_f##
defm SVFCVT_F64_F16   : SInstCvtMXZ<"svcvt_f64[_f16]", "ddPO", "dPO", "d", "aarch64_sve_fcvt_f64f16">;
defm SVFCVT_F64_F32   : SInstCvtMXZ<"svcvt_f64[_f32]", "ddPM", "dPM", "d", "aarch64_sve_fcvt_f64f32">;

let ArchGuard = "defined(__ARM_FEATURE_SVE2)" in {
defm SVCVTLT_F32    : SInstCvtMX<"svcvtlt_f32[_f16]",  "ddPh", "dPh", "f", "aarch64_sve_fcvtlt_f32f16">;
defm SVCVTLT_F64    : SInstCvtMX<"svcvtlt_f64[_f32]",  "ddPh", "dPh", "d", "aarch64_sve_fcvtlt_f64f32">;

defm SVCVTX_F32     : SInstCvtMXZ<"svcvtx_f32[_f64]",  "MMPd", "MPd", "d", "aarch64_sve_fcvtx_f32f64">;

def SVCVTNT_F32     : SInst<"svcvtnt_f16[_f32]",  "hhPd", "f", MergeOp1, "aarch64_sve_fcvtnt_f16f32", [IsOverloadNone]>;
def SVCVTNT_F64     : SInst<"svcvtnt_f32[_f64]",  "hhPd", "d", MergeOp1, "aarch64_sve_fcvtnt_f32f64", [IsOverloadNone]>;
//  SVCVTNT_X       : Implemented as macro by SveEmitter.cpp

def SVCVTXNT_F32    : SInst<"svcvtxnt_f32[_f64]", "MMPd", "d", MergeOp1, "aarch64_sve_fcvtxnt_f32f64", [IsOverloadNone]>;
//  SVCVTXNT_X_F32  : Implemented as macro by SveEmitter.cpp

}

def SVCADD_M : SInst<"svcadd[_{d}]", "dPddi",  "hfd", MergeOp1,  "aarch64_sve_fcadd", [], [ImmCheck<3, ImmCheckComplexRot90_270>]>;
def SVCMLA_M : SInst<"svcmla[_{d}]", "dPdddi", "hfd", MergeOp1,  "aarch64_sve_fcmla", [], [ImmCheck<4, ImmCheckComplexRotAll90>]>;

////////////////////////////////////////////////////////////////////////////////
// Saturating scalar arithmetic
def SVQDECH_S : SInst<"svqdech_pat[_{d}]",   "ddIi", "s", MergeNone, "aarch64_sve_sqdech", [], [ImmCheck<2, ImmCheck1_16>]>;
def SVQDECH_U : SInst<"svqdech_pat[_{d}]",   "ddIi", "Us", MergeNone, "aarch64_sve_uqdech", [], [ImmCheck<2, ImmCheck1_16>]>;


////////////////////////////////////////////////////////////////////////////////
// Predicate creation

def SVPFALSE : SInst<"svpfalse[_b]", "P", "", MergeNone, "", [IsOverloadNone]>;


////////////////////////////////////////////////////////////////////////////////
// Counting elements

def SVCNTB_PAT : SInst<"svcntb_pat", "nI", "", MergeNone, "aarch64_sve_cntb", [IsOverloadNone]>;
def SVCNTH_PAT : SInst<"svcnth_pat", "nI", "", MergeNone, "aarch64_sve_cnth", [IsOverloadNone]>;
def SVCNTW_PAT : SInst<"svcntw_pat", "nI", "", MergeNone, "aarch64_sve_cntw", [IsOverloadNone]>;
def SVCNTD_PAT : SInst<"svcntd_pat", "nI", "", MergeNone, "aarch64_sve_cntd", [IsOverloadNone]>;
////////////////////////////////////////////////////////////////////////////////
// Integer arithmetic
def SVDOT_LANE_S : SInst<"svdot_lane[_{d}]",  "ddqqi",  "il",   MergeNone, "aarch64_sve_sdot_lane", [], [ImmCheck<3, ImmCheckLaneIndexDot, 2>]>;
def SVDOT_LANE_U : SInst<"svdot_lane[_{d}]",  "ddqqi",  "UiUl", MergeNone, "aarch64_sve_udot_lane", [], [ImmCheck<3, ImmCheckLaneIndexDot, 2>]>;

////////////////////////////////////////////////////////////////////////////////
// SVE2 WhileGE/GT
let ArchGuard = "defined(__ARM_FEATURE_SVE2)" in {
def SVWHILEGE_S32 : SInst<"svwhilege_{d}[_{1}]", "Pkk", "PcPsPiPl",     MergeNone, "aarch64_sve_whilege", [IsOverloadWhile]>;
def SVWHILEGE_S64 : SInst<"svwhilege_{d}[_{1}]", "Pll", "PcPsPiPl",     MergeNone, "aarch64_sve_whilege", [IsOverloadWhile]>;
def SVWHILEGT_S32 : SInst<"svwhilegt_{d}[_{1}]", "Pkk", "PcPsPiPl",     MergeNone, "aarch64_sve_whilegt", [IsOverloadWhile]>;
def SVWHILEGT_S64 : SInst<"svwhilegt_{d}[_{1}]", "Pll", "PcPsPiPl",     MergeNone, "aarch64_sve_whilegt", [IsOverloadWhile]>;
def SVWHILEHI_U32 : SInst<"svwhilegt_{d}[_{1}]", "Pmm", "PUcPUsPUiPUl", MergeNone, "aarch64_sve_whilehi", [IsOverloadWhile]>;
def SVWHILEHI_U64 : SInst<"svwhilegt_{d}[_{1}]", "Pnn", "PUcPUsPUiPUl", MergeNone, "aarch64_sve_whilehi", [IsOverloadWhile]>;
def SVWHILEHS_U32 : SInst<"svwhilege_{d}[_{1}]", "Pmm", "PUcPUsPUiPUl", MergeNone, "aarch64_sve_whilehs", [IsOverloadWhile]>;
def SVWHILEHS_U64 : SInst<"svwhilege_{d}[_{1}]", "Pnn", "PUcPUsPUiPUl", MergeNone, "aarch64_sve_whilehs", [IsOverloadWhile]>;
}

////////////////////////////////////////////////////////////////////////////////
// SVE2 - Non-temporal gather/scatter
let ArchGuard = "defined(__ARM_FEATURE_SVE2)" in {
// Non-temporal gather load one vector (vector base)
def SVLDNT1_GATHER_BASES_U   : MInst<"svldnt1_gather[_{2}base]_{0}",   "dPu", "ilUiUlfd", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1SB_GATHER_BASES_U : MInst<"svldnt1sb_gather[_{2}base]_{0}", "dPu", "ilUiUl",   [IsGatherLoad],               MemEltTyInt8,    "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1UB_GATHER_BASES_U : MInst<"svldnt1ub_gather[_{2}base]_{0}", "dPu", "ilUiUl",   [IsGatherLoad, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1SH_GATHER_BASES_U : MInst<"svldnt1sh_gather[_{2}base]_{0}", "dPu", "ilUiUl",   [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1UH_GATHER_BASES_U : MInst<"svldnt1uh_gather[_{2}base]_{0}", "dPu", "ilUiUl",   [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1SW_GATHER_BASES_U : MInst<"svldnt1sw_gather[_{2}base]_{0}", "dPu", "lUl",      [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1UW_GATHER_BASES_U : MInst<"svldnt1uw_gather[_{2}base]_{0}", "dPu", "lUl",      [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldnt1_gather_scalar_offset">;

// Non-temporal gather load one vector (scalar base, signed vector offset in bytes)
def SVLDNT1_GATHER_64B_OFFSETS_S   : MInst<"svldnt1_gather_[{3}]offset[_{0}]", "dPcx", "lUld", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ldnt1_gather">;
def SVLDNT1SB_GATHER_64B_OFFSETS_S : MInst<"svldnt1sb_gather_[{3}]offset_{0}", "dPSx", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ldnt1_gather">;
def SVLDNT1UB_GATHER_64B_OFFSETS_S : MInst<"svldnt1ub_gather_[{3}]offset_{0}", "dPWx", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldnt1_gather">;
def SVLDNT1SH_GATHER_64B_OFFSETS_S : MInst<"svldnt1sh_gather_[{3}]offset_{0}", "dPTx", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ldnt1_gather">;
def SVLDNT1UH_GATHER_64B_OFFSETS_S : MInst<"svldnt1uh_gather_[{3}]offset_{0}", "dPXx", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnt1_gather">;
def SVLDNT1SW_GATHER_64B_OFFSETS_S : MInst<"svldnt1sw_gather_[{3}]offset_{0}", "dPUx", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt32,   "aarch64_sve_ldnt1_gather">;
def SVLDNT1UW_GATHER_64B_OFFSETS_S : MInst<"svldnt1uw_gather_[{3}]offset_{0}", "dPYx", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldnt1_gather">;

// Non-temporal gather load one vector (scalar base, unsigned vector offset in bytes)
def SVLDNT1_GATHER_64B_OFFSETS_U   : MInst<"svldnt1_gather_[{3}]offset[_{0}]", "dPcu", "lUld", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ldnt1_gather">;
def SVLDNT1SB_GATHER_64B_OFFSETS_U : MInst<"svldnt1sb_gather_[{3}]offset_{0}", "dPSu", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ldnt1_gather">;
def SVLDNT1UB_GATHER_64B_OFFSETS_U : MInst<"svldnt1ub_gather_[{3}]offset_{0}", "dPWu", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldnt1_gather">;
def SVLDNT1SH_GATHER_64B_OFFSETS_U : MInst<"svldnt1sh_gather_[{3}]offset_{0}", "dPTu", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ldnt1_gather">;
def SVLDNT1UH_GATHER_64B_OFFSETS_U : MInst<"svldnt1uh_gather_[{3}]offset_{0}", "dPXu", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnt1_gather">;
def SVLDNT1SW_GATHER_64B_OFFSETS_U : MInst<"svldnt1sw_gather_[{3}]offset_{0}", "dPUu", "lUl",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt32,   "aarch64_sve_ldnt1_gather">;
def SVLDNT1UW_GATHER_64B_OFFSETS_U : MInst<"svldnt1uw_gather_[{3}]offset_{0}", "dPYu", "lUl",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldnt1_gather">;

def SVLDNT1_GATHER_32B_OFFSETS_U   : MInst<"svldnt1_gather_[{3}]offset[_{0}]", "dPcu", "iUif", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ldnt1_gather_uxtw">;
def SVLDNT1SB_GATHER_32B_OFFSETS_U : MInst<"svldnt1sb_gather_[{3}]offset_{0}", "dPSu", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ldnt1_gather_uxtw">;
def SVLDNT1UB_GATHER_32B_OFFSETS_U : MInst<"svldnt1ub_gather_[{3}]offset_{0}", "dPWu", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldnt1_gather_uxtw">;
def SVLDNT1SH_GATHER_32B_OFFSETS_U : MInst<"svldnt1sh_gather_[{3}]offset_{0}", "dPTu", "iUi",  [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ldnt1_gather_uxtw">;
def SVLDNT1UH_GATHER_32B_OFFSETS_U : MInst<"svldnt1uh_gather_[{3}]offset_{0}", "dPXu", "iUi",  [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnt1_gather_uxtw">;

// Non-temporal gather load one vector (vector base, scalar offset in bytes)
def SVLDNT1_GATHER_OFFSET_S   : MInst<"svldnt1_gather[_{2}base]_offset_{0}",   "dPul", "ilUiUlfd", [IsGatherLoad, IsByteIndexed],               MemEltTyDefault, "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1SB_GATHER_OFFSET_S : MInst<"svldnt1sb_gather[_{2}base]_offset_{0}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed],               MemEltTyInt8,    "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1UB_GATHER_OFFSET_S : MInst<"svldnt1ub_gather[_{2}base]_offset_{0}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt8,    "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1SH_GATHER_OFFSET_S : MInst<"svldnt1sh_gather[_{2}base]_offset_{0}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed],               MemEltTyInt16,   "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1UH_GATHER_OFFSET_S : MInst<"svldnt1uh_gather[_{2}base]_offset_{0}", "dPul", "ilUiUl",   [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1SW_GATHER_OFFSET_S : MInst<"svldnt1sw_gather[_{2}base]_offset_{0}", "dPul", "lUl",      [IsGatherLoad, IsByteIndexed],               MemEltTyInt32,   "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1UW_GATHER_OFFSET_S : MInst<"svldnt1uw_gather[_{2}base]_offset_{0}", "dPul", "lUl",      [IsGatherLoad, IsByteIndexed, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldnt1_gather_scalar_offset">;

// Non-temporal gather load one vector (scalar base, signed vector index)
def SVLDNT1_GATHER_64B_INDICES_S   : MInst<"svldnt1_gather_[{3}]index[_{0}]", "dPcx", "lUld", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldnt1_gather_index">;
def SVLDNT1SH_GATHER_64B_INDICES_S : MInst<"svldnt1sh_gather_[{3}]index_{0}", "dPTx", "lUl",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldnt1_gather_index">;
def SVLDNT1UH_GATHER_64B_INDICES_S : MInst<"svldnt1uh_gather_[{3}]index_{0}", "dPXx", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnt1_gather_index">;
def SVLDNT1SW_GATHER_64B_INDICES_S : MInst<"svldnt1sw_gather_[{3}]index_{0}", "dPUx", "lUl",  [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ldnt1_gather_index">;
def SVLDNT1UW_GATHER_64B_INDICES_S : MInst<"svldnt1uw_gather_[{3}]index_{0}", "dPYx", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldnt1_gather_index">;

// Non temporal gather load one vector (scalar base, unsigned vector index)
def SVLDNT1_GATHER_64B_INDICES_U   : MInst<"svldnt1_gather_[{3}]index[_{0}]", "dPcu", "lUld", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldnt1_gather_index">;
def SVLDNT1SH_GATHER_64B_INDICES_U : MInst<"svldnt1sh_gather_[{3}]index_{0}", "dPTu", "lUl",  [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldnt1_gather_index">;
def SVLDNT1UH_GATHER_64B_INDICES_U : MInst<"svldnt1uh_gather_[{3}]index_{0}", "dPXu", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnt1_gather_index">;
def SVLDNT1SW_GATHER_64B_INDICES_U : MInst<"svldnt1sw_gather_[{3}]index_{0}", "dPUu", "lUl",  [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ldnt1_gather_index">;
def SVLDNT1UW_GATHER_64B_INDICES_U : MInst<"svldnt1uw_gather_[{3}]index_{0}", "dPYu", "lUl",  [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldnt1_gather_index">;

// Non-temporal gather load one vector (vector base, signed scalar index)
def SVLDNT1_GATHER_INDEX_S   : MInst<"svldnt1_gather[_{2}base]_index_{0}",   "dPul", "ilUiUlfd", [IsGatherLoad],               MemEltTyDefault, "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1SH_GATHER_INDEX_S : MInst<"svldnt1sh_gather[_{2}base]_index_{0}", "dPul", "ilUiUl",   [IsGatherLoad],               MemEltTyInt16,   "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1UH_GATHER_INDEX_S : MInst<"svldnt1uh_gather[_{2}base]_index_{0}", "dPul", "ilUiUl",   [IsGatherLoad, IsZExtReturn], MemEltTyInt16,   "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1SW_GATHER_INDEX_S : MInst<"svldnt1sw_gather[_{2}base]_index_{0}", "dPul", "lUl",      [IsGatherLoad],               MemEltTyInt32,   "aarch64_sve_ldnt1_gather_scalar_offset">;
def SVLDNT1UW_GATHER_INDEX_S : MInst<"svldnt1uw_gather[_{2}base]_index_{0}", "dPul", "lUl",      [IsGatherLoad, IsZExtReturn], MemEltTyInt32,   "aarch64_sve_ldnt1_gather_scalar_offset">;

// Non-temporal scatter store one vector (vector base)
def SVSTNT1_SCATTER_BASES_U  : MInst<"svstnt1_scatter[_{2}base_{d}]",  "vPud", "ilUiUlfd", [IsScatterStore], MemEltTyDefault, "aarch64_sve_stnt1_scatter_scalar_offset">;
def SVSTNT1B_SCATTER_BASES_U : MInst<"svstnt1b_scatter[_{2}base_{d}]", "vPud", "ilUiUl",   [IsScatterStore], MemEltTyInt8,    "aarch64_sve_stnt1_scatter_scalar_offset">;
def SVSTNT1H_SCATTER_BASES_U : MInst<"svstnt1h_scatter[_{2}base_{d}]", "vPud", "ilUiUl",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_stnt1_scatter_scalar_offset">;
def SVSTNT1W_SCATTER_BASES_U : MInst<"svstnt1w_scatter[_{2}base_{d}]", "vPud", "lUl",      [IsScatterStore], MemEltTyInt32,   "aarch64_sve_stnt1_scatter_scalar_offset">;

// Non-temporal scatter store one vector (scalar base, signed vector offset in bytes)
def SVSTNT1_SCATTER_64B_OFFSETS_S   : MInst<"svstnt1_scatter_[{3}]offset[_{d}]",  "vPpxd", "lUld", [IsScatterStore, IsByteIndexed], MemEltTyDefault, "aarch64_sve_stnt1_scatter">;
def SVSTNT1B_SCATTER_64B_OFFSETS_SS : MInst<"svstnt1b_scatter_[{3}]offset[_{d}]", "vPAxd", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_stnt1_scatter">;
def SVSTNT1B_SCATTER_64B_OFFSETS_SU : MInst<"svstnt1b_scatter_[{3}]offset[_{d}]", "vPExd", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_stnt1_scatter">;
def SVSTNT1H_SCATTER_64B_OFFSETS_SS : MInst<"svstnt1h_scatter_[{3}]offset[_{d}]", "vPBxd", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_stnt1_scatter">;
def SVSTNT1H_SCATTER_64B_OFFSETS_SU : MInst<"svstnt1h_scatter_[{3}]offset[_{d}]", "vPFxd", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_stnt1_scatter">;
def SVSTNT1W_SCATTER_64B_OFFSETS_SS : MInst<"svstnt1w_scatter_[{3}]offset[_{d}]", "vPCxd", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_stnt1_scatter">;
def SVSTNT1W_SCATTER_64B_OFFSETS_SU : MInst<"svstnt1w_scatter_[{3}]offset[_{d}]", "vPGxd", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_stnt1_scatter">;

// Non-temporal scatter store one vector (scalar base, unsigned vector offset in bytes)
def SVSTNT1_SCATTER_64B_OFFSETS_U   : MInst<"svstnt1_scatter_[{3}]offset[_{d}]",  "vPpud", "lUld", [IsScatterStore, IsByteIndexed], MemEltTyDefault, "aarch64_sve_stnt1_scatter">;
def SVSTNT1B_SCATTER_64B_OFFSETS_US : MInst<"svstnt1b_scatter_[{3}]offset[_{d}]", "vPAud", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_stnt1_scatter">;
def SVSTNT1B_SCATTER_64B_OFFSETS_UU : MInst<"svstnt1b_scatter_[{3}]offset[_{d}]", "vPEud", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_stnt1_scatter">;
def SVSTNT1H_SCATTER_64B_OFFSETS_US : MInst<"svstnt1h_scatter_[{3}]offset[_{d}]", "vPBud", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_stnt1_scatter">;
def SVSTNT1H_SCATTER_64B_OFFSETS_UU : MInst<"svstnt1h_scatter_[{3}]offset[_{d}]", "vPFud", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_stnt1_scatter">;
def SVSTNT1W_SCATTER_64B_OFFSETS_US : MInst<"svstnt1w_scatter_[{3}]offset[_{d}]", "vPCud", "l",    [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_stnt1_scatter">;
def SVSTNT1W_SCATTER_64B_OFFSETS_UU : MInst<"svstnt1w_scatter_[{3}]offset[_{d}]", "vPGud", "Ul",   [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_stnt1_scatter">;

def SVSTNT1_SCATTER_32B_OFFSETS_U   : MInst<"svstnt1_scatter_[{3}]offset[_{d}]",  "vPpud", "iUif", [IsScatterStore, IsByteIndexed], MemEltTyDefault, "aarch64_sve_stnt1_scatter_uxtw">;
def SVSTNT1B_SCATTER_32B_OFFSETS_US : MInst<"svstnt1b_scatter_[{3}]offset[_{d}]", "vPAud", "i",    [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_stnt1_scatter_uxtw">;
def SVSTNT1B_SCATTER_32B_OFFSETS_UU : MInst<"svstnt1b_scatter_[{3}]offset[_{d}]", "vPEud", "Ui",   [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_stnt1_scatter_uxtw">;
def SVSTNT1H_SCATTER_32B_OFFSETS_US : MInst<"svstnt1h_scatter_[{3}]offset[_{d}]", "vPBud", "i",    [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_stnt1_scatter_uxtw">;
def SVSTNT1H_SCATTER_32B_OFFSETS_UU : MInst<"svstnt1h_scatter_[{3}]offset[_{d}]", "vPFud", "Ui",   [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_stnt1_scatter_uxtw">;

// Non-temporal scatter store one vector (vector base, scalar offset in bytes)
def SVSTNT1_SCATTER_OFFSET_S  : MInst<"svstnt1_scatter[_{2}base]_offset[_{d}]",  "vPuld", "ilUiUlfd", [IsScatterStore, IsByteIndexed], MemEltTyDefault, "aarch64_sve_stnt1_scatter_scalar_offset">;
def SVSTNT1B_SCATTER_OFFSET_S : MInst<"svstnt1b_scatter[_{2}base]_offset[_{d}]", "vPuld", "ilUiUl",   [IsScatterStore, IsByteIndexed], MemEltTyInt8,    "aarch64_sve_stnt1_scatter_scalar_offset">;
def SVSTNT1H_SCATTER_OFFSET_S : MInst<"svstnt1h_scatter[_{2}base]_offset[_{d}]", "vPuld", "ilUiUl",   [IsScatterStore, IsByteIndexed], MemEltTyInt16,   "aarch64_sve_stnt1_scatter_scalar_offset">;
def SVSTNT1W_SCATTER_OFFSET_S : MInst<"svstnt1w_scatter[_{2}base]_offset[_{d}]", "vPuld", "lUl",      [IsScatterStore, IsByteIndexed], MemEltTyInt32,   "aarch64_sve_stnt1_scatter_scalar_offset">;

// Non-temporal scatter store one vector (scalar base, signed vector index)
def SVSTNT1_SCATTER_INDICES_S   : MInst<"svstnt1_scatter_[{3}]index[_{d}]",  "vPpxd", "lUld", [IsScatterStore], MemEltTyDefault, "aarch64_sve_stnt1_scatter_index">;
def SVSTNT1H_SCATTER_INDICES_SS : MInst<"svstnt1h_scatter_[{3}]index[_{d}]", "vPBxd", "l",    [IsScatterStore], MemEltTyInt16,   "aarch64_sve_stnt1_scatter_index">;
def SVSTNT1H_SCATTER_INDICES_SU : MInst<"svstnt1h_scatter_[{3}]index[_{d}]", "vPFxd", "Ul",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_stnt1_scatter_index">;
def SVSTNT1W_SCATTER_INDICES_SS : MInst<"svstnt1w_scatter_[{3}]index[_{d}]", "vPCxd", "l",    [IsScatterStore], MemEltTyInt32,   "aarch64_sve_stnt1_scatter_index">;
def SVSTNT1W_SCATTER_INDICES_SU : MInst<"svstnt1w_scatter_[{3}]index[_{d}]", "vPGxd", "Ul",   [IsScatterStore], MemEltTyInt32,   "aarch64_sve_stnt1_scatter_index">;

// Non-temporal scatter store one vector (scalar base, unsigned vector index)
def SVSTNT1_SCATTER_INDICES_U   : MInst<"svstnt1_scatter_[{3}]index[_{d}]",  "vPpud", "lUld", [IsScatterStore], MemEltTyDefault, "aarch64_sve_stnt1_scatter_index">;
def SVSTNT1H_SCATTER_INDICES_US : MInst<"svstnt1h_scatter_[{3}]index[_{d}]", "vPBud", "l",    [IsScatterStore], MemEltTyInt16,   "aarch64_sve_stnt1_scatter_index">;
def SVSTNT1H_SCATTER_INDICES_UU : MInst<"svstnt1h_scatter_[{3}]index[_{d}]", "vPFud", "Ul",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_stnt1_scatter_index">;
def SVSTNT1W_SCATTER_INDICES_US : MInst<"svstnt1w_scatter_[{3}]index[_{d}]", "vPCud", "l",    [IsScatterStore], MemEltTyInt32,   "aarch64_sve_stnt1_scatter_index">;
def SVSTNT1W_SCATTER_INDICES_UU : MInst<"svstnt1w_scatter_[{3}]index[_{d}]", "vPGud", "Ul",   [IsScatterStore], MemEltTyInt32,   "aarch64_sve_stnt1_scatter_index">;

// Non-temporal scatter store one vector (vector base, signed scalar index)
def SVSTNT1_SCATTER_INDEX_S  : MInst<"svstnt1_scatter[_{2}base]_index[_{d}]",  "vPuld", "ilUiUlfd", [IsScatterStore], MemEltTyDefault, "aarch64_sve_stnt1_scatter_scalar_offset">;
def SVSTNT1H_SCATTER_INDEX_S : MInst<"svstnt1h_scatter[_{2}base]_index[_{d}]", "vPuld", "ilUiUl",   [IsScatterStore], MemEltTyInt16,   "aarch64_sve_stnt1_scatter_scalar_offset">;
def SVSTNT1W_SCATTER_INDEX_S : MInst<"svstnt1w_scatter[_{2}base]_index[_{d}]", "vPuld", "lUl",      [IsScatterStore], MemEltTyInt32,   "aarch64_sve_stnt1_scatter_scalar_offset">;
}

////////////////////////////////////////////////////////////////////////////////
// SVE2 - Contiguous conflict detection
let ArchGuard = "defined(__ARM_FEATURE_SVE2)" in {
def SVWHILERW_B : SInst<"svwhilerw[_{1}]", "Pcc", "cUc",  MergeNone, "aarch64_sve_whilerw_b", [IsOverloadWhileRW]>;
def SVWHILERW_H : SInst<"svwhilerw[_{1}]", "Pcc", "sUsh", MergeNone, "aarch64_sve_whilerw_h", [IsOverloadWhileRW]>;
def SVWHILERW_S : SInst<"svwhilerw[_{1}]", "Pcc", "iUif", MergeNone, "aarch64_sve_whilerw_s", [IsOverloadWhileRW]>;
def SVWHILERW_D : SInst<"svwhilerw[_{1}]", "Pcc", "lUld", MergeNone, "aarch64_sve_whilerw_d", [IsOverloadWhileRW]>;

def SVWHILEWR_B : SInst<"svwhilewr[_{1}]", "Pcc", "cUc",  MergeNone, "aarch64_sve_whilewr_b", [IsOverloadWhileRW]>;
def SVWHILEWR_H : SInst<"svwhilewr[_{1}]", "Pcc", "sUsh", MergeNone, "aarch64_sve_whilewr_h", [IsOverloadWhileRW]>;
def SVWHILEWR_S : SInst<"svwhilewr[_{1}]", "Pcc", "iUif", MergeNone, "aarch64_sve_whilewr_s", [IsOverloadWhileRW]>;
def SVWHILEWR_D : SInst<"svwhilewr[_{1}]", "Pcc", "lUld", MergeNone, "aarch64_sve_whilewr_d", [IsOverloadWhileRW]>;
}
