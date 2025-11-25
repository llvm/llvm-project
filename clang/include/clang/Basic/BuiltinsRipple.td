//===--- BuiltinsRipple.td - Ripple Builtin function defs -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Ripple builtin function database.
//
//===----------------------------------------------------------------------===//

/// Ripple get Processing unit Index.
def RippleGetPuIndex : Builtin {
  let Spellings = ["__builtin_ripple_get_index"];
  let Attributes = [NoThrow];

/// We cannot use custom structure pointer types here; we check that the void
/// pointer actually is a pointer to a struct ripple_block_shape as a semantic
/// check in SemaRipple::CheckBuiltinFunctionCall
  let Prototype = "size_t(void*, _Constant size_t)";
}

def RippleGetParallelIndex : Builtin {
  let Spellings = ["__builtin_ripple_parallel_idx"];
  let Attributes = [NoThrow];
  let Prototype = "size_t(void*, _Constant size_t, ...)";
}

def RippleGetSize : Builtin {
  let Spellings = ["__builtin_ripple_get_size"];
  let Attributes = [NoThrow];
  let Prototype = "size_t(void*, _Constant size_t)";
}

def RippleSetBlockShape : Builtin {
  let Spellings = ["__builtin_ripple_set_shape"];
  let Attributes = [NoThrow];
  let Prototype = "void*(_Constant size_t,"
                  "_Constant size_t, _Constant size_t,"
                  "_Constant size_t, _Constant size_t,"
                  "_Constant size_t, _Constant size_t,"
                  "_Constant size_t, _Constant size_t,"
                  "_Constant size_t, _Constant size_t)";
}

class RippleIntTypesTemplate : Template<
  ["char", "unsigned char", "short", "unsigned short", "int32_t", "uint32_t", "int64_t",
  "uint64_t"],
  ["i8",    "u8",           "i16",   "u16",             "i32",     "u32",      "i64",
  "u64"     ]>;

class RippleAllTypesTemplate : Template<
  ["char", "unsigned char", "short", "unsigned short", "int32_t", "uint32_t", "int64_t",
   "uint64_t", "_Float16", "__bf16", "float", "double"],
  ["i8",    "u8",           "i16",   "u16",             "i32",     "u32",      "i64",
   "u64",      "f16",      "bf16",   "f32",   "f64"   ]>;

def RippleReduceAdd : Builtin, RippleAllTypesTemplate {
  let Spellings = ["__builtin_ripple_reduceadd_"];
  let Attributes = [NoThrow];
  let Prototype = "T(uint64_t, T)";
}

def RippleReduceMax : Builtin, RippleAllTypesTemplate {
  let Spellings = ["__builtin_ripple_reducemax_"];
  let Attributes = [NoThrow];
  let Prototype = "T(uint64_t, T)";
}

def RippleReduceMin : Builtin, RippleAllTypesTemplate {
  let Spellings = ["__builtin_ripple_reducemin_"];
  let Attributes = [NoThrow];
  let Prototype = "T(uint64_t, T)";
}

class RippleReduceFloatTypesTemplate : Template<
  ["_Float16", "__bf16", "float", "double"],
  ["f16", "bf16", "f32", "f64"]>;

def RippleReduceMaximum : Builtin, RippleReduceFloatTypesTemplate {
  let Spellings = ["__builtin_ripple_reducemaximum_"];
  let Attributes = [NoThrow];
  let Prototype = "T(uint64_t, T)";
}

def RippleReduceMinimum : Builtin, RippleReduceFloatTypesTemplate {
  let Spellings = ["__builtin_ripple_reduceminimum_"];
  let Attributes = [NoThrow];
  let Prototype = "T(uint64_t, T)";
}

def RippleReduceAnd : Builtin, RippleIntTypesTemplate {
  let Spellings = ["__builtin_ripple_reduceand_"];
  let Attributes = [NoThrow];
  let Prototype = "T(uint64_t, T)";
}

def RippleReduceOr : Builtin, RippleIntTypesTemplate {
  let Spellings = ["__builtin_ripple_reduceor_"];
  let Attributes = [NoThrow];
  let Prototype = "T(uint64_t, T)";
}

// FIXME: Don't see how to allow a function-type to enter here.
def RippleShuffle : Builtin, RippleAllTypesTemplate {
  let Spellings = ["__builtin_ripple_shuffle_"];
  let Attributes = [NoThrow];
  let Prototype = "T(T, T, bool, ...)";
}

class RippleSaturatedOpBuiltin<string name> : Builtin, RippleIntTypesTemplate {
  let Spellings = ["__builtin_ripple_" # name # "_sat_"];
  let Attributes = [NoThrow, ConstIgnoringErrnoAndExceptions];
  let Prototype = "T(T, T)";
}

def RippleAddSat : RippleSaturatedOpBuiltin<"add">;
def RippleSubSat : RippleSaturatedOpBuiltin<"sub">;
def RippleShlSat : RippleSaturatedOpBuiltin<"shl">;

class RippleFloatTypesTemplate
    : Template<["_Float16", "__bf16", "float", "double", "long double"],
               ["f16", "bf16", "f", "", "l"]>;

class RippleMathBuiltin<string name> : Builtin, RippleFloatTypesTemplate {
  let Spellings = ["__builtin_ripple_" #name];
  let Attributes = [NoThrow, ConstIgnoringErrnoAndExceptions];
}
class RippleUnaryMathBuiltin<string name> : RippleMathBuiltin<name> {
  let Prototype = "T(T)";
}
class RippleBinaryMathBuiltin<string name> : RippleMathBuiltin<name> {
  let Prototype = "T(T, T)";
}

// Follow the order of llvm intrinsics to better spot the differences
def RippleSqrt : RippleUnaryMathBuiltin<"sqrt">;
def RippleASin : RippleUnaryMathBuiltin<"asin">;
def RippleACos : RippleUnaryMathBuiltin<"acos">;
def RippleATan : RippleUnaryMathBuiltin<"atan">;
def RippleATan2 : RippleBinaryMathBuiltin<"atan2">;
def RippleSin : RippleUnaryMathBuiltin<"sin">;
def RippleCos : RippleUnaryMathBuiltin<"cos">;
def RippleTan : RippleUnaryMathBuiltin<"tan">;
def RippleSinh : RippleUnaryMathBuiltin<"sinh">;
def RippleCosh : RippleUnaryMathBuiltin<"cosh">;
def RippleTanh : RippleUnaryMathBuiltin<"tanh">;
def RipplePow : RippleBinaryMathBuiltin<"pow">;
def RippleLog : RippleUnaryMathBuiltin<"log">;
def RippleLog10 : RippleUnaryMathBuiltin<"log10">;
def RippleLog2 : RippleUnaryMathBuiltin<"log2">;
def RippleExp : RippleUnaryMathBuiltin<"exp">;
def RippleExp2 : RippleUnaryMathBuiltin<"exp2">;
def RippleExp10 : RippleUnaryMathBuiltin<"exp10">;
def RippleFabs : RippleUnaryMathBuiltin<"fabs">;
def RippleCopysign : RippleBinaryMathBuiltin<"copysign">;
def RippleFloor : RippleUnaryMathBuiltin<"floor">;
def RippleCeil : RippleUnaryMathBuiltin<"ceil">;
def RippleTrunc : RippleUnaryMathBuiltin<"trunc">;
def RippleRint : RippleUnaryMathBuiltin<"rint">;
def RippleNearbyint : RippleUnaryMathBuiltin<"nearbyint">;
def RippleRound : RippleUnaryMathBuiltin<"round">;
def RippleRoundeven : RippleUnaryMathBuiltin<"roundeven">;
def RippleSincos : RippleMathBuiltin<"sincos"> { let Prototype = "void(T, T*, T*)"; }
def RippleModf : RippleMathBuiltin<"modf"> { let Prototype = "T(T, T*)"; }
def RippleLdexp : RippleMathBuiltin<"ldexp"> { let Prototype = "T(T, int)"; }
def RippleFrexp : RippleMathBuiltin<"frexp"> { let Prototype = "T(T, int*)"; }

class RippleAllTypesAndPtrTemplate : Template<
  ["char", "unsigned char", "short", "unsigned short", "int32_t", "uint32_t", "int64_t",
   "uint64_t", "_Float16", "__bf16", "float", "double", "void*"],
  ["i8",    "u8",           "i16",   "u16",             "i32",     "u32",      "i64",
   "u64",      "f16",      "bf16",   "f32",   "f64",  "p" ]>;

def RippleBroadcast : Builtin, RippleAllTypesAndPtrTemplate {
  let Spellings = ["__builtin_ripple_broadcast_"];
  let Attributes = [NoThrow];
  let Prototype = "T(void*, uint64_t, T)";
}

def RippleSlice : Builtin, RippleAllTypesAndPtrTemplate {
  let Spellings = ["__builtin_ripple_slice_"];
  let Attributes = [NoThrow];
  let Prototype = "T(T, ...)";
}
