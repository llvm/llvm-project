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
// d: default
// c: const pointer type
// P: predicate type

class MergeType<int val> {
  int Value = val;
}
def MergeNone    : MergeType<0>;
def MergeAny     : MergeType<1>;
def MergeOp1     : MergeType<2>;
def MergeZero    : MergeType<3>;
def MergeAnyExp  : MergeType<4>; // Use merged builtin with explicit
def MergeZeroExp : MergeType<5>; // generation of its inactive argument.

class MemEltTy<int val> {
  int Value = val;
}
def MemEltTyDefault   : MemEltTy<0>;
def MemEltTyInt8      : MemEltTy<1>;
def MemEltTyInt16     : MemEltTy<2>;
def MemEltTyInt32     : MemEltTy<3>;
def MemEltTyInt64     : MemEltTy<4>;

class FlagType<int val> {
  int Value = val;
}

// These must be kept in sync with the flags in utils/TableGen/SveEmitter.h
// and include/clang/Basic/TargetBuiltins.h
def NoFlags                   : FlagType<0x00000000>;
//                            0x00000001 => EltType
//                            ...
//                            0x0000000f => EltType
//                            0x00000010 => MemEltType
//                            ...
//                            0x00000070 => MemEltType
def IsLoad                    : FlagType<0x00000080>;

// Every intrinsic subclasses Inst.
class Inst<string n, string p, string t, MergeType mt, string i,
           list<FlagType> ft, MemEltTy met> {
  string Name = n;
  string Prototype = p;
  string Types = t;
  string ArchGuard = "";
  int Merge = mt.Value;
  string LLVMIntrinsic = i;
  list<FlagType> Flags = ft;
  int MemEltType = met.Value;
}

// MInst: Instructions which access memory
class MInst<string n, string p, string t, list<FlagType> f,
            MemEltTy met=MemEltTyDefault, string i="">
  : Inst<n, p, t, MergeNone, i, f, met> {}

////////////////////////////////////////////////////////////////////////////////
// Loads

// Load one vector (scalar base)
def SVLD1   : MInst<"svld1[_{2}]", "dPc", "csilUcUsUiUlhfd", [IsLoad]>;
