//==--- OpenCLBuiltins.td - OpenCL builtin declarations -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains TableGen definitions for OpenCL builtin function
// declarations.  In case of an unresolved function name in OpenCL, Clang will
// check for a function described in this file when -fdeclare-opencl-builtins
// is specified.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//              Definitions of miscellaneous basic entities.
//===----------------------------------------------------------------------===//
// Versions of OpenCL
class Version<int _Version> {
  int ID = _Version;
}
def CLAll : Version<  0>;
def CL10  : Version<100>;
def CL11  : Version<110>;
def CL12  : Version<120>;
def CL20  : Version<200>;

// Address spaces
// Pointer types need to be assigned an address space.
class AddressSpace<string _AS> {
  string Name = _AS;
}
def DefaultAS    : AddressSpace<"clang::LangAS::Default">;
def PrivateAS    : AddressSpace<"clang::LangAS::opencl_private">;
def GlobalAS     : AddressSpace<"clang::LangAS::opencl_global">;
def ConstantAS   : AddressSpace<"clang::LangAS::opencl_constant">;
def LocalAS      : AddressSpace<"clang::LangAS::opencl_local">;
def GenericAS    : AddressSpace<"clang::LangAS::opencl_generic">;

// OpenCL language extension.
class AbstractExtension<string _Ext> {
  // One or more OpenCL extensions, space separated.  Each extension must be
  // a valid extension name for the opencl extension pragma.
  string ExtName = _Ext;
}

// Extension associated to a builtin function.
class FunctionExtension<string _Ext> : AbstractExtension<_Ext>;

// FunctionExtension definitions.
def FuncExtNone                          : FunctionExtension<"">;
def FuncExtKhrSubgroups                  : FunctionExtension<"cl_khr_subgroups">;
def FuncExtKhrGlobalInt32BaseAtomics     : FunctionExtension<"cl_khr_global_int32_base_atomics">;
def FuncExtKhrGlobalInt32ExtendedAtomics : FunctionExtension<"cl_khr_global_int32_extended_atomics">;
def FuncExtKhrLocalInt32BaseAtomics      : FunctionExtension<"cl_khr_local_int32_base_atomics">;
def FuncExtKhrLocalInt32ExtendedAtomics  : FunctionExtension<"cl_khr_local_int32_extended_atomics">;
def FuncExtKhrInt64BaseAtomics           : FunctionExtension<"cl_khr_int64_base_atomics">;
def FuncExtKhrInt64ExtendedAtomics       : FunctionExtension<"cl_khr_int64_extended_atomics">;
def FuncExtKhrMipmapImage                : FunctionExtension<"cl_khr_mipmap_image">;
def FuncExtKhrGlMsaaSharing              : FunctionExtension<"cl_khr_gl_msaa_sharing">;

// Multiple extensions
def FuncExtKhrMipmapAndWrite3d           : FunctionExtension<"cl_khr_mipmap_image cl_khr_3d_image_writes">;

// Qualified Type.  These map to ASTContext::QualType.
class QualType<string _Name, bit _IsAbstract=0> {
  // Name of the field or function in a clang::ASTContext
  // E.g. Name="IntTy" for the int type, and "getIntPtrType()" for an intptr_t
  string Name = _Name;
  // Some QualTypes in this file represent an abstract type for which there is
  // no corresponding AST QualType, e.g. a GenType or an `image2d_t` type
  // without access qualifiers.
  bit IsAbstract = _IsAbstract;
}

// List of integers.
class IntList<string _Name, list<int> _List> {
  string Name = _Name;
  list<int> List = _List;
}

//===----------------------------------------------------------------------===//
//                      OpenCL C classes for types
//===----------------------------------------------------------------------===//
// OpenCL C basic data types (int, float, image2d_t, ...).
// Its child classes can represent concrete types (e.g. VectorType) or
// abstract types (e.g. GenType).
class Type<string _Name, QualType _QTName> {
  // Name of the Type.
  string Name = _Name;
  // QualType associated with this type.
  QualType QTName = _QTName;
  // Size of the vector (if applicable).
  int VecWidth = 1;
  // Is a pointer.
  bit IsPointer = 0;
  // "const" qualifier.
  bit IsConst = 0;
  // "volatile" qualifier.
  bit IsVolatile = 0;
  // Access qualifier. Must be one of ("RO", "WO", "RW").
  string AccessQualifier = "";
  // Address space.
  string AddrSpace = DefaultAS.Name;
}

// OpenCL vector types (e.g. int2, int3, int16, float8, ...).
class VectorType<Type _Ty, int _VecWidth> : Type<_Ty.Name, _Ty.QTName> {
  let VecWidth = _VecWidth;
  let AccessQualifier = "";
  // Inherited fields
  let IsPointer = _Ty.IsPointer;
  let IsConst = _Ty.IsConst;
  let IsVolatile = _Ty.IsVolatile;
  let AddrSpace = _Ty.AddrSpace;
}

// OpenCL pointer types (e.g. int*, float*, ...).
class PointerType<Type _Ty, AddressSpace _AS = DefaultAS> :
                                      Type<_Ty.Name, _Ty.QTName> {
  let AddrSpace = _AS.Name;
  // Inherited fields
  let VecWidth = _Ty.VecWidth;
  let IsPointer = 1;
  let IsConst = _Ty.IsConst;
  let IsVolatile = _Ty.IsVolatile;
  let AccessQualifier = _Ty.AccessQualifier;
}

// OpenCL const types (e.g. const int).
class ConstType<Type _Ty> : Type<_Ty.Name, _Ty.QTName> {
  let IsConst = 1;
  // Inherited fields
  let VecWidth = _Ty.VecWidth;
  let IsPointer = _Ty.IsPointer;
  let IsVolatile = _Ty.IsVolatile;
  let AccessQualifier = _Ty.AccessQualifier;
  let AddrSpace = _Ty.AddrSpace;
}

// OpenCL volatile types (e.g. volatile int).
class VolatileType<Type _Ty> : Type<_Ty.Name, _Ty.QTName> {
  let IsVolatile = 1;
  // Inherited fields
  let VecWidth = _Ty.VecWidth;
  let IsPointer = _Ty.IsPointer;
  let IsConst = _Ty.IsConst;
  let AccessQualifier = _Ty.AccessQualifier;
  let AddrSpace = _Ty.AddrSpace;
}

// OpenCL image types (e.g. image2d).
class ImageType<Type _Ty, string _AccessQualifier> :
          Type<_Ty.Name, QualType<_Ty.QTName.Name#_AccessQualifier#"Ty", 0>> {
  let VecWidth = 0;
  let AccessQualifier = _AccessQualifier;
  // Inherited fields
  let IsPointer = _Ty.IsPointer;
  let IsConst = _Ty.IsConst;
  let IsVolatile = _Ty.IsVolatile;
  let AddrSpace = _Ty.AddrSpace;
}

// List of Types.
class TypeList<string _Name, list<Type> _Type> {
  string Name = _Name;
  list<Type> List = _Type;
}

// A GenericType is an abstract type that defines a set of types as a
// combination of Types and vector sizes.
//
// For example, if TypeList = <int, float> and VectorList = <1, 2, 4>, then it
// represents <int, int2, int4, float, float2, float4>.
//
// Some rules apply when using multiple GenericType arguments in a declaration:
//   1. The number of vector sizes must be equal or 1 for all gentypes in a
//      declaration.
//   2. The number of Types must be equal or 1 for all gentypes in a
//      declaration.
//   3. Generic types are combined by iterating over all generic types at once.
//      For example, for the following GenericTypes
//        GenT1 = GenericType<half, [1, 2]> and
//        GenT2 = GenericType<float, int, [1, 2]>
//      A declaration f(GenT1, GenT2) results in the combinations
//        f(half, float), f(half2, float2), f(half, int), f(half2, int2) .
//   4. "sgentype" from the OpenCL specification is supported by specifying
//      a single vector size.
//      For example, for the following GenericTypes
//        GenT = GenericType<half, int, [1, 2]> and
//        SGenT = GenericType<half, int, [1]>
//      A declaration f(GenT, SGenT) results in the combinations
//        f(half, half), f(half2, half), f(int, int), f(int2, int) .
class GenericType<string _Ty, TypeList _TypeList, IntList _VectorList> :
                                            Type<_Ty, QualType<"null", 1>> {
  // Possible element types of the generic type.
  TypeList TypeList = _TypeList;
  // Possible vector sizes of the types in the TypeList.
  IntList VectorList = _VectorList;
  // The VecWidth field is ignored for GenericTypes. Use VectorList instead.
  let VecWidth = 0;
}

// Builtin function attributes.
def Attr {
  list<bit> None = [0, 0, 0];
  list<bit> Pure = [1, 0, 0];
  list<bit> Const = [0, 1, 0];
  list<bit> Convergent = [0, 0, 1];
}

//===----------------------------------------------------------------------===//
//                      OpenCL C class for builtin functions
//===----------------------------------------------------------------------===//
class Builtin<string _Name, list<Type> _Signature, list<bit> _Attributes = Attr.None> {
  // Name of the builtin function
  string Name = _Name;
  // List of types used by the function. The first one is the return type and
  // the following are the arguments. The list must have at least one element
  // (the return type).
  list<Type> Signature = _Signature;
  // Function attribute __attribute__((pure))
  bit IsPure = _Attributes[0];
  // Function attribute __attribute__((const))
  bit IsConst = _Attributes[1];
  // Function attribute __attribute__((convergent))
  bit IsConv = _Attributes[2];
  // OpenCL extensions to which the function belongs.
  FunctionExtension Extension = FuncExtNone;
  // Version of OpenCL from which the function is available (e.g.: CL10).
  // MinVersion is inclusive.
  Version MinVersion = CL10;
  // Version of OpenCL from which the function is not supported anymore.
  // MaxVersion is exclusive.
  // CLAll makes the function available for all versions.
  Version MaxVersion = CLAll;
}

//===----------------------------------------------------------------------===//
//                 Definitions of OpenCL C types
//===----------------------------------------------------------------------===//

// OpenCL v1.0/1.2/2.0 s6.1.1: Built-in Scalar Data Types.
def Bool      : Type<"bool",      QualType<"BoolTy">>;
def Char      : Type<"char",      QualType<"CharTy">>;
def UChar     : Type<"uchar",     QualType<"UnsignedCharTy">>;
def Short     : Type<"short",     QualType<"ShortTy">>;
def UShort    : Type<"ushort",    QualType<"UnsignedShortTy">>;
def Int       : Type<"int",       QualType<"IntTy">>;
def UInt      : Type<"uint",      QualType<"UnsignedIntTy">>;
def Long      : Type<"long",      QualType<"LongTy">>;
def ULong     : Type<"ulong",     QualType<"UnsignedLongTy">>;
def Float     : Type<"float",     QualType<"FloatTy">>;
def Double    : Type<"double",    QualType<"DoubleTy">>;
def Half      : Type<"half",      QualType<"HalfTy">>;
def Size      : Type<"size_t",    QualType<"getSizeType()">>;
def PtrDiff   : Type<"ptrdiff_t", QualType<"getPointerDiffType()">>;
def IntPtr    : Type<"intptr_t",  QualType<"getIntPtrType()">>;
def UIntPtr   : Type<"uintPtr_t", QualType<"getUIntPtrType()">>;
def Void      : Type<"void_t",    QualType<"VoidTy">>;

// OpenCL v1.0/1.2/2.0 s6.1.2: Built-in Vector Data Types.
// Built-in vector data types are created by TableGen's OpenCLBuiltinEmitter.

// OpenCL v1.0/1.2/2.0 s6.1.3: Other Built-in Data Types.
// The image definitions are "abstract".  They should not be used without
// specifying an access qualifier (RO/WO/RW).
def Image1d               : Type<"Image1d", QualType<"OCLImage1d", 1>>;
def Image2d               : Type<"Image2d", QualType<"OCLImage2d", 1>>;
def Image3d               : Type<"Image3d", QualType<"OCLImage3d", 1>>;
def Image1dArray          : Type<"Image1dArray", QualType<"OCLImage1dArray", 1>>;
def Image1dBuffer         : Type<"Image1dBuffer", QualType<"OCLImage1dBuffer", 1>>;
def Image2dArray          : Type<"Image2dArray", QualType<"OCLImage2dArray", 1>>;
def Image2dDepth          : Type<"Image2dDepth", QualType<"OCLImage2dDepth", 1>>;
def Image2dArrayDepth     : Type<"Image2dArrayDepth", QualType<"OCLImage2dArrayDepth", 1>>;
def Image2dMsaa           : Type<"Image2dMsaa", QualType<"OCLImage2dMSAA", 1>>;
def Image2dArrayMsaa      : Type<"Image2dArrayMsaa", QualType<"OCLImage2dArrayMSAA", 1>>;
def Image2dMsaaDepth      : Type<"Image2dMsaaDepth", QualType<"OCLImage2dMSAADepth", 1>>;
def Image2dArrayMsaaDepth : Type<"Image2dArrayMsaaDepth", QualType<"OCLImage2dArrayMSAADepth", 1>>;

def Sampler           : Type<"Sampler", QualType<"OCLSamplerTy">>;
def Event             : Type<"Event", QualType<"OCLEventTy">>;

//===----------------------------------------------------------------------===//
//                 Definitions of OpenCL gentype variants
//===----------------------------------------------------------------------===//
// The OpenCL specification often uses "gentype" in builtin function
// declarations to indicate that a builtin function is available with various
// argument and return types.  The types represented by "gentype" vary between
// different parts of the specification.  The following definitions capture
// the different type lists for gentypes in different parts of the
// specification.

// Vector width lists.
def VecAndScalar: IntList<"VecAndScalar", [1, 2, 3, 4, 8, 16]>;
def VecNoScalar : IntList<"VecNoScalar", [2, 3, 4, 8, 16]>;
def Vec1        : IntList<"Vec1", [1]>;
def Vec2        : IntList<"Vec2", [2]>;
def Vec4        : IntList<"Vec4", [4]>;
def Vec8        : IntList<"Vec8", [8]>;
def Vec16       : IntList<"Vec16", [16]>;
def Vec1234     : IntList<"Vec1234", [1, 2, 3, 4]>;

// Type lists.
def TLAll         : TypeList<"TLAll", [Char,  UChar, Short,  UShort, Int,  UInt, Long,  ULong, Float, Double, Half]>;
def TLAllUnsigned : TypeList<"TLAllUnsigned", [UChar, UChar, UShort, UShort, UInt, UInt, ULong, ULong, UInt,  ULong,  UShort]>;
def TLFloat : TypeList<"TLFloat", [Float, Double, Half]>;
def TLSignedInts   : TypeList<"TLSignedInts", [Char, Short, Int, Long]>;
def TLUnsignedInts : TypeList<"TLUnsignedInts", [UChar, UShort, UInt, ULong]>;

def TLIntLongFloats : TypeList<"TLIntLongFloats", [Int, UInt, Long, ULong, Float, Double, Half]>;

// All unsigned integer types twice, to facilitate unsigned return types for e.g.
// uchar abs(char) and
// uchar abs(uchar).
def TLAllUIntsTwice : TypeList<"TLAllUIntsTwice", [UChar, UChar, UShort, UShort, UInt, UInt, ULong, ULong]>;

def TLAllInts : TypeList<"TLAllInts", [Char, UChar, Short, UShort, Int, UInt, Long, ULong]>;

// GenType definitions for multiple base types (e.g. all floating point types,
// or all integer types).
// All types
def AGenTypeN              : GenericType<"AGenTypeN", TLAll, VecAndScalar>;
def AGenTypeNNoScalar      : GenericType<"AGenTypeNNoScalar", TLAll, VecNoScalar>;
// All integer
def AIGenType1             : GenericType<"AIGenType1", TLAllInts, Vec1>;
def AIGenTypeN             : GenericType<"AIGenTypeN", TLAllInts, VecAndScalar>;
def AIGenTypeNNoScalar     : GenericType<"AIGenTypeNNoScalar", TLAllInts, VecNoScalar>;
// All integer to unsigned
def AI2UGenTypeN           : GenericType<"AI2UGenTypeN", TLAllUIntsTwice, VecAndScalar>;
// Signed integer
def SGenTypeN              : GenericType<"SGenTypeN", TLSignedInts, VecAndScalar>;
// Unsigned integer
def UGenTypeN              : GenericType<"UGenTypeN", TLUnsignedInts, VecAndScalar>;
// Float
def FGenTypeN              : GenericType<"FGenTypeN", TLFloat, VecAndScalar>;
// (u)int, (u)long, and all floats
def IntLongFloatGenType1   : GenericType<"IntLongFloatGenType1", TLIntLongFloats, Vec1>;

// GenType definitions for every single base type (e.g. fp32 only).
// Names are like: GenTypeFloatVecAndScalar.
foreach Type = [Char, UChar, Short, UShort,
                Int, UInt, Long, ULong,
                Float, Double, Half] in {
  foreach VecSizes = [VecAndScalar, VecNoScalar] in {
    def "GenType" # Type # VecSizes :
              GenericType<"GenType" # Type # VecSizes,
                          TypeList<"GL" # Type.Name, [Type]>,
                          VecSizes>;
  }
}

// GenType definitions for vec1234.
foreach Type = [Float, Double, Half] in {
  def "GenType" # Type # Vec1234 :
              GenericType<"GenType" # Type # Vec1234,
                          TypeList<"GL" # Type.Name, [Type]>,
                          Vec1234>;
}


//===----------------------------------------------------------------------===//
//                 Definitions of OpenCL builtin functions
//===----------------------------------------------------------------------===//
//--------------------------------------------------------------------
// OpenCL v1.1/1.2/2.0 s6.2.3 - Explicit conversions.
// OpenCL v2.0 Extensions s5.1.1 and s6.1.1 - Conversions.

// Generate the convert_* builtins functions.
foreach RType = [Float, Double, Half, Char, UChar, Short,
                 UShort, Int, UInt, Long, ULong] in {
  foreach IType = [Float, Double, Half, Char, UChar, Short,
                   UShort, Int, UInt, Long, ULong] in {
    foreach sat = ["", "_sat"] in {
      foreach rnd = ["", "_rte", "_rtn", "_rtp", "_rtz"] in {
        def : Builtin<"convert_" # RType.Name # sat # rnd, [RType, IType],
                      Attr.Const>;
        foreach v = [2, 3, 4, 8, 16] in {
          def : Builtin<"convert_" # RType.Name # v # sat # rnd,
                        [VectorType<RType, v>, VectorType<IType, v>],
                        Attr.Const>;
        }
      }
    }
  }
}

//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.1, v1.2 s6.12.1, v2.0 s6.13.1 - Work-item Functions
// --- Table 7 ---
def : Builtin<"get_work_dim", [UInt], Attr.Const>;
foreach name = ["get_global_size", "get_global_id", "get_local_size",
                "get_local_id", "get_num_groups", "get_group_id",
                "get_global_offset"] in {
  def : Builtin<name, [Size, UInt], Attr.Const>;
}

let MinVersion = CL20 in {
  def : Builtin<"get_enqueued_local_size", [Size, UInt]>;
  foreach name = ["get_global_linear_id", "get_local_linear_id"] in {
    def : Builtin<name, [Size]>;
  }
}


//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.2, v1.2 s6.12.2, v2.0 s6.13.2 - Math functions
// OpenCL Extension v2.0 s5.1.2 and s6.1.2 - Math Functions
// --- Table 8 ---
// --- 1 argument ---
foreach name = ["acos", "acosh", "acospi",
                "asin", "asinh", "asinpi",
                "atan", "atanh", "atanpi",
                "cbrt", "ceil",
                "cos", "cosh", "cospi",
                "erfc", "erf",
                "exp", "exp2", "exp10", "expm1",
                "fabs", "floor",
                "log", "log2", "log10", "log1p", "logb",
                "rint", "round", "rsqrt",
                "sin", "sinh", "sinpi",
                "sqrt",
                "tan", "tanh", "tanpi",
                "tgamma", "trunc",
                "lgamma"] in {
    def : Builtin<name, [FGenTypeN, FGenTypeN], Attr.Const>;
}
foreach name = ["nan"] in {
  def : Builtin<name, [GenTypeFloatVecAndScalar, GenTypeUIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecAndScalar, GenTypeULongVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecAndScalar, GenTypeUShortVecAndScalar], Attr.Const>;
}

// --- 2 arguments ---
foreach name = ["atan2", "atan2pi", "copysign", "fdim", "fmod", "hypot",
                "maxmag", "minmag", "nextafter", "pow", "powr",
                "remainder"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN, FGenTypeN], Attr.Const>;
}
foreach name = ["fmax", "fmin"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN, FGenTypeN], Attr.Const>;
  def : Builtin<name, [GenTypeFloatVecNoScalar, GenTypeFloatVecNoScalar, Float], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecNoScalar, GenTypeDoubleVecNoScalar, Double], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecNoScalar, GenTypeHalfVecNoScalar, Half], Attr.Const>;
}
foreach name = ["ilogb"] in {
  def : Builtin<name, [GenTypeIntVecAndScalar, GenTypeFloatVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeIntVecAndScalar, GenTypeDoubleVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeIntVecAndScalar, GenTypeHalfVecAndScalar], Attr.Const>;
}
foreach name = ["ldexp"] in {
  def : Builtin<name, [GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar, GenTypeIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeFloatVecNoScalar, GenTypeFloatVecNoScalar, Int], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecNoScalar, GenTypeDoubleVecNoScalar, Int], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecAndScalar, GenTypeHalfVecAndScalar, GenTypeIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecNoScalar, GenTypeHalfVecNoScalar, Int], Attr.Const>;
}
foreach name = ["pown", "rootn"] in {
  def : Builtin<name, [GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar, GenTypeIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecAndScalar, GenTypeHalfVecAndScalar, GenTypeIntVecAndScalar], Attr.Const>;
}

// --- 3 arguments ---
foreach name = ["fma", "mad"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN, FGenTypeN, FGenTypeN], Attr.Const>;
}

// --- Version dependent ---
let MaxVersion = CL20 in {
  foreach AS = [GlobalAS, LocalAS, PrivateAS] in {
    foreach name = ["fract", "modf", "sincos"] in {
      def : Builtin<name, [FGenTypeN, FGenTypeN, PointerType<FGenTypeN, AS>]>;
    }
    foreach name = ["frexp", "lgamma_r"] in {
      foreach Type = [GenTypeFloatVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeHalfVecAndScalar] in {
        def : Builtin<name, [Type, Type, PointerType<GenTypeIntVecAndScalar, AS>]>;
      }
    }
    foreach name = ["remquo"] in {
      foreach Type = [GenTypeFloatVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeHalfVecAndScalar] in {
        def : Builtin<name, [Type, Type, Type, PointerType<GenTypeIntVecAndScalar, AS>]>;
      }
    }
  }
}
let MinVersion = CL20 in {
  foreach name = ["fract", "modf", "sincos"] in {
    def : Builtin<name, [FGenTypeN, FGenTypeN, PointerType<FGenTypeN, GenericAS>]>;
  }
  foreach name = ["frexp", "lgamma_r"] in {
    foreach Type = [GenTypeFloatVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeHalfVecAndScalar] in {
      def : Builtin<name, [Type, Type, PointerType<GenTypeIntVecAndScalar, GenericAS>]>;
    }  }
  foreach name = ["remquo"] in {
    foreach Type = [GenTypeFloatVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeHalfVecAndScalar] in {
      def : Builtin<name, [Type, Type, Type, PointerType<GenTypeIntVecAndScalar, GenericAS>]>;
    }
  }
}

// --- Table 9 ---
foreach name = ["half_cos",
                "half_exp", "half_exp2", "half_exp10",
                "half_log", "half_log2", "half_log10",
                "half_recip", "half_rsqrt",
                "half_sin", "half_sqrt", "half_tan",
                "native_cos",
                "native_exp", "native_exp2", "native_exp10",
                "native_log", "native_log2", "native_log10",
                "native_recip", "native_rsqrt",
                "native_sin", "native_sqrt", "native_tan"] in {
  def : Builtin<name, [GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar], Attr.Const>;
}
foreach name = ["half_divide", "half_powr",
                "native_divide", "native_powr"] in {
  def : Builtin<name, [GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar], Attr.Const>;
}

//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.3, v1.2 s6.12.3, v2.0 s6.13.3 - Integer Functions
// --- Table 10 ---
// --- 1 argument ---
foreach name = ["abs"] in {
  def : Builtin<name, [AI2UGenTypeN, AIGenTypeN], Attr.Const>;
}
foreach name = ["clz", "popcount"] in {
  def : Builtin<name, [AIGenTypeN, AIGenTypeN], Attr.Const>;
}
let MinVersion = CL20 in {
  foreach name = ["ctz"] in {
    def : Builtin<name, [AIGenTypeN, AIGenTypeN]>;
  }
}

// --- 2 arguments ---
foreach name = ["abs_diff"] in {
  def : Builtin<name, [AI2UGenTypeN, AIGenTypeN, AIGenTypeN], Attr.Const>;
}
foreach name = ["add_sat", "hadd", "rhadd", "mul_hi", "rotate", "sub_sat"] in {
  def : Builtin<name, [AIGenTypeN, AIGenTypeN, AIGenTypeN], Attr.Const>;
}
foreach name = ["max", "min"] in {
  def : Builtin<name, [AIGenTypeN, AIGenTypeN, AIGenTypeN], Attr.Const>;
  def : Builtin<name, [AIGenTypeNNoScalar, AIGenTypeNNoScalar, AIGenType1], Attr.Const>;
}
foreach name = ["upsample"] in {
  def : Builtin<name, [GenTypeShortVecAndScalar, GenTypeCharVecAndScalar, GenTypeUCharVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeUShortVecAndScalar, GenTypeUCharVecAndScalar, GenTypeUCharVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeIntVecAndScalar, GenTypeShortVecAndScalar, GenTypeUShortVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeUIntVecAndScalar, GenTypeUShortVecAndScalar, GenTypeUShortVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeLongVecAndScalar, GenTypeIntVecAndScalar, GenTypeUIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeULongVecAndScalar, GenTypeUIntVecAndScalar, GenTypeUIntVecAndScalar], Attr.Const>;
}

// --- 3 arguments ---
foreach name = ["clamp"] in {
  def : Builtin<name, [AIGenTypeN, AIGenTypeN, AIGenTypeN, AIGenTypeN], Attr.Const>;
  def : Builtin<name, [AIGenTypeNNoScalar, AIGenTypeNNoScalar, AIGenType1, AIGenType1], Attr.Const>;
}
foreach name = ["mad_hi", "mad_sat"] in {
  def : Builtin<name, [AIGenTypeN, AIGenTypeN, AIGenTypeN, AIGenTypeN], Attr.Const>;
}

// --- Table 11 ---
foreach name = ["mad24"] in {
  def : Builtin<name, [GenTypeIntVecAndScalar, GenTypeIntVecAndScalar, GenTypeIntVecAndScalar, GenTypeIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeUIntVecAndScalar, GenTypeUIntVecAndScalar, GenTypeUIntVecAndScalar, GenTypeUIntVecAndScalar], Attr.Const>;
}
foreach name = ["mul24"] in {
  def : Builtin<name, [GenTypeIntVecAndScalar, GenTypeIntVecAndScalar, GenTypeIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeUIntVecAndScalar, GenTypeUIntVecAndScalar, GenTypeUIntVecAndScalar], Attr.Const>;
}

//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.4, v1.2 s6.12.4, v2.0 s6.13.4 - Common Functions
// OpenCL Extension v2.0 s5.1.3 and s6.1.3 - Common Functions
// --- Table 12 ---
// --- 1 argument ---
foreach name = ["degrees", "radians", "sign"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN], Attr.Const>;
}

// --- 2 arguments ---
foreach name = ["max", "min"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN, FGenTypeN], Attr.Const>;
  def : Builtin<name, [GenTypeFloatVecNoScalar, GenTypeFloatVecNoScalar, Float], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecNoScalar, GenTypeDoubleVecNoScalar, Double], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecNoScalar, GenTypeHalfVecNoScalar, Half], Attr.Const>;
}
foreach name = ["step"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN, FGenTypeN], Attr.Const>;
  def : Builtin<name, [GenTypeFloatVecNoScalar, Float, GenTypeFloatVecNoScalar], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecNoScalar, Double, GenTypeDoubleVecNoScalar], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecNoScalar, Half, GenTypeHalfVecNoScalar], Attr.Const>;
}

// --- 3 arguments ---
foreach name = ["clamp", "mix"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN, FGenTypeN, FGenTypeN], Attr.Const>;
  def : Builtin<name, [GenTypeFloatVecNoScalar, GenTypeFloatVecNoScalar, Float, Float], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecNoScalar, GenTypeDoubleVecNoScalar, Double, Double], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecNoScalar, GenTypeHalfVecNoScalar, Half, Half], Attr.Const>;
}
foreach name = ["smoothstep"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN, FGenTypeN, FGenTypeN], Attr.Const>;
  def : Builtin<name, [GenTypeFloatVecNoScalar, Float, Float, GenTypeFloatVecNoScalar], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecNoScalar, Double, Double, GenTypeDoubleVecNoScalar], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecNoScalar, Half, Half, GenTypeHalfVecNoScalar], Attr.Const>;
}


//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.5, v1.2 s6.12.5, v2.0 s6.13.5 - Geometric Functions
// OpenCL Extension v2.0 s5.1.4 and s6.1.4 - Geometric Functions
// --- Table 13 ---
// --- 1 argument ---
foreach name = ["length"] in {
  def : Builtin<name, [Float, GenTypeFloatVec1234], Attr.Const>;
  def : Builtin<name, [Double, GenTypeDoubleVec1234], Attr.Const>;
  def : Builtin<name, [Half, GenTypeHalfVec1234], Attr.Const>;
}
foreach name = ["normalize"] in {
  def : Builtin<name, [GenTypeFloatVec1234, GenTypeFloatVec1234], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVec1234, GenTypeDoubleVec1234], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVec1234, GenTypeHalfVec1234], Attr.Const>;
}
foreach name = ["fast_length"] in {
  def : Builtin<name, [Float, GenTypeFloatVec1234], Attr.Const>;
}
foreach name = ["fast_normalize"] in {
  def : Builtin<name, [GenTypeFloatVec1234, GenTypeFloatVec1234], Attr.Const>;
}

// --- 2 arguments ---
foreach name = ["cross"] in {
  foreach VSize = [3, 4] in {
    def : Builtin<name, [VectorType<Float, VSize>, VectorType<Float, VSize>, VectorType<Float, VSize>], Attr.Const>;
    def : Builtin<name, [VectorType<Double, VSize>, VectorType<Double, VSize>, VectorType<Double, VSize>], Attr.Const>;
    def : Builtin<name, [VectorType<Half, VSize>, VectorType<Half, VSize>, VectorType<Half, VSize>], Attr.Const>;
  }
}
foreach name = ["dot", "distance"] in {
  def : Builtin<name, [Float, GenTypeFloatVec1234, GenTypeFloatVec1234], Attr.Const>;
  def : Builtin<name, [Double, GenTypeDoubleVec1234, GenTypeDoubleVec1234], Attr.Const>;
  def : Builtin<name, [Half, GenTypeHalfVec1234, GenTypeHalfVec1234], Attr.Const>;
}
foreach name = ["fast_distance"] in {
  def : Builtin<name, [Float, GenTypeFloatVec1234, GenTypeFloatVec1234], Attr.Const>;
}


//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.6, v1.2 s6.12.6, v2.0 s6.13.6 - Relational Functions
// OpenCL Extension v2.0 s5.1.5 and s6.1.5 - Relational Functions
// --- Table 14 ---
// --- 1 argument ---
foreach name = ["isfinite", "isinf", "isnan", "isnormal", "signbit"] in {
  def : Builtin<name, [GenTypeIntVecAndScalar, GenTypeFloatVecAndScalar], Attr.Const>;
  def : Builtin<name, [Int, Double], Attr.Const>;
  def : Builtin<name, [GenTypeLongVecNoScalar, GenTypeDoubleVecNoScalar], Attr.Const>;
  def : Builtin<name, [Int, Half], Attr.Const>;
  def : Builtin<name, [GenTypeShortVecNoScalar, GenTypeHalfVecNoScalar], Attr.Const>;
}
foreach name = ["any", "all"] in {
  def : Builtin<name, [Int, AIGenTypeN], Attr.Const>;
}

// --- 2 arguments ---
foreach name = ["isequal", "isnotequal", "isgreater", "isgreaterequal",
                "isless", "islessequal", "islessgreater", "isordered",
                "isunordered"] in {
  def : Builtin<name, [GenTypeIntVecAndScalar, GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar], Attr.Const>;
  def : Builtin<name, [Int, Double, Double], Attr.Const>;
  def : Builtin<name, [GenTypeLongVecNoScalar, GenTypeDoubleVecNoScalar, GenTypeDoubleVecNoScalar], Attr.Const>;
  def : Builtin<name, [Int, Half, Half], Attr.Const>;
  def : Builtin<name, [GenTypeShortVecNoScalar, GenTypeHalfVecNoScalar, GenTypeHalfVecNoScalar], Attr.Const>;
}

// --- 3 arguments ---
foreach name = ["bitselect"] in {
  def : Builtin<name, [AGenTypeN, AGenTypeN, AGenTypeN, AGenTypeN], Attr.Const>;
}
foreach name = ["select"] in {
  def : Builtin<name, [SGenTypeN, SGenTypeN, SGenTypeN, SGenTypeN], Attr.Const>;
  def : Builtin<name, [SGenTypeN, SGenTypeN, SGenTypeN, UGenTypeN], Attr.Const>;
  def : Builtin<name, [UGenTypeN, UGenTypeN, UGenTypeN, UGenTypeN], Attr.Const>;
  def : Builtin<name, [UGenTypeN, UGenTypeN, UGenTypeN, SGenTypeN], Attr.Const>;
  def : Builtin<name, [GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar, GenTypeIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar, GenTypeFloatVecAndScalar, GenTypeUIntVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeLongVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeDoubleVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeDoubleVecAndScalar, GenTypeULongVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecAndScalar, GenTypeHalfVecAndScalar, GenTypeHalfVecAndScalar, GenTypeShortVecAndScalar], Attr.Const>;
  def : Builtin<name, [GenTypeHalfVecAndScalar, GenTypeHalfVecAndScalar, GenTypeHalfVecAndScalar, GenTypeUShortVecAndScalar], Attr.Const>;
}


//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.7, v1.2 s6.12.7, v2.0 s6.13.7 - Vector Data Load and Store Functions
// OpenCL Extension v1.1 s9.3.6 and s9.6.6, v1.2 s9.5.6, v2.0 s5.1.6 and s6.1.6 - Vector Data Load and Store Functions
// --- Table 15 ---
// Variants for OpenCL versions below 2.0, using pointers to the global, local
// and private address spaces.
let MaxVersion = CL20 in {
  foreach AS = [GlobalAS, LocalAS, PrivateAS] in {
    foreach VSize = [2, 3, 4, 8, 16] in {
      foreach name = ["vload" # VSize] in {
        def : Builtin<name, [VectorType<Char, VSize>, Size, PointerType<ConstType<Char>, AS>]>;
        def : Builtin<name, [VectorType<UChar, VSize>, Size, PointerType<ConstType<UChar>, AS>]>;
        def : Builtin<name, [VectorType<Short, VSize>, Size, PointerType<ConstType<Short>, AS>]>;
        def : Builtin<name, [VectorType<UShort, VSize>, Size, PointerType<ConstType<UShort>, AS>]>;
        def : Builtin<name, [VectorType<Int, VSize>, Size, PointerType<ConstType<Int>, AS>]>;
        def : Builtin<name, [VectorType<UInt, VSize>, Size, PointerType<ConstType<UInt>, AS>]>;
        def : Builtin<name, [VectorType<Long, VSize>, Size, PointerType<ConstType<Long>, AS>]>;
        def : Builtin<name, [VectorType<ULong, VSize>, Size, PointerType<ConstType<ULong>, AS>]>;
        def : Builtin<name, [VectorType<Float, VSize>, Size, PointerType<ConstType<Float>, AS>]>;
        def : Builtin<name, [VectorType<Double, VSize>, Size, PointerType<ConstType<Double>, AS>]>;
        def : Builtin<name, [VectorType<Half, VSize>, Size, PointerType<ConstType<Half>, AS>]>;
      }
      foreach name = ["vstore" # VSize] in {
        def : Builtin<name, [Void, VectorType<Char, VSize>, Size, PointerType<ConstType<Char>, AS>]>;
        def : Builtin<name, [Void, VectorType<UChar, VSize>, Size, PointerType<ConstType<UChar>, AS>]>;
        def : Builtin<name, [Void, VectorType<Short, VSize>, Size, PointerType<ConstType<Short>, AS>]>;
        def : Builtin<name, [Void, VectorType<UShort, VSize>, Size, PointerType<ConstType<UShort>, AS>]>;
        def : Builtin<name, [Void, VectorType<Int, VSize>, Size, PointerType<ConstType<Int>, AS>]>;
        def : Builtin<name, [Void, VectorType<UInt, VSize>, Size, PointerType<ConstType<UInt>, AS>]>;
        def : Builtin<name, [Void, VectorType<Long, VSize>, Size, PointerType<ConstType<Long>, AS>]>;
        def : Builtin<name, [Void, VectorType<ULong, VSize>, Size, PointerType<ConstType<ULong>, AS>]>;
        def : Builtin<name, [Void, VectorType<Float, VSize>, Size, PointerType<ConstType<Float>, AS>]>;
        def : Builtin<name, [Void, VectorType<Double, VSize>, Size, PointerType<ConstType<Double>, AS>]>;
        def : Builtin<name, [Void, VectorType<Half, VSize>, Size, PointerType<ConstType<Half>, AS>]>;
      }
      foreach name = ["vloada_half" # VSize] in {
        def : Builtin<name, [VectorType<Float, VSize>, Size, PointerType<ConstType<Half>, AS>]>;
      }
      foreach rnd = ["", "_rte", "_rtz", "_rtp", "_rtn"] in {
        foreach name = ["vstorea_half" # VSize # rnd] in {
          def : Builtin<name, [Void, VectorType<Float, VSize>, Size, PointerType<Half, AS>]>;
          def : Builtin<name, [Void, VectorType<Double, VSize>, Size, PointerType<Half, AS>]>;
        }
      }
    }
  }
}
// Variants for OpenCL versions above 2.0, using pointers to the generic
// address space.
let MinVersion = CL20 in {
  foreach VSize = [2, 3, 4, 8, 16] in {
    foreach name = ["vload" # VSize] in {
      def : Builtin<name, [VectorType<Char, VSize>, Size, PointerType<ConstType<Char>, GenericAS>]>;
      def : Builtin<name, [VectorType<UChar, VSize>, Size, PointerType<ConstType<UChar>, GenericAS>]>;
      def : Builtin<name, [VectorType<Short, VSize>, Size, PointerType<ConstType<Short>, GenericAS>]>;
      def : Builtin<name, [VectorType<UShort, VSize>, Size, PointerType<ConstType<UShort>, GenericAS>]>;
      def : Builtin<name, [VectorType<Int, VSize>, Size, PointerType<ConstType<Int>, GenericAS>]>;
      def : Builtin<name, [VectorType<UInt, VSize>, Size, PointerType<ConstType<UInt>, GenericAS>]>;
      def : Builtin<name, [VectorType<Long, VSize>, Size, PointerType<ConstType<Long>, GenericAS>]>;
      def : Builtin<name, [VectorType<ULong, VSize>, Size, PointerType<ConstType<ULong>, GenericAS>]>;
      def : Builtin<name, [VectorType<Float, VSize>, Size, PointerType<ConstType<Float>, GenericAS>]>;
      def : Builtin<name, [VectorType<Double, VSize>, Size, PointerType<ConstType<Double>, GenericAS>]>;
      def : Builtin<name, [VectorType<Half, VSize>, Size, PointerType<ConstType<Half>, GenericAS>]>;
    }
    foreach name = ["vstore" # VSize] in {
      def : Builtin<name, [Void, VectorType<Char, VSize>, Size, PointerType<ConstType<Char>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<UChar, VSize>, Size, PointerType<ConstType<UChar>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<Short, VSize>, Size, PointerType<ConstType<Short>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<UShort, VSize>, Size, PointerType<ConstType<UShort>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<Int, VSize>, Size, PointerType<ConstType<Int>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<UInt, VSize>, Size, PointerType<ConstType<UInt>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<Long, VSize>, Size, PointerType<ConstType<Long>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<ULong, VSize>, Size, PointerType<ConstType<ULong>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<Float, VSize>, Size, PointerType<ConstType<Float>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<Double, VSize>, Size, PointerType<ConstType<Double>, GenericAS>]>;
      def : Builtin<name, [Void, VectorType<Half, VSize>, Size, PointerType<ConstType<Half>, GenericAS>]>;
    }
    foreach name = ["vloada_half" # VSize] in {
      def : Builtin<name, [VectorType<Float, VSize>, Size, PointerType<ConstType<Half>, GenericAS>]>;
    }
    foreach rnd = ["", "_rte", "_rtz", "_rtp", "_rtn"] in {
      foreach name = ["vstorea_half" # VSize # rnd] in {
        def : Builtin<name, [Void, VectorType<Float, VSize>, Size, PointerType<Half, GenericAS>]>;
        def : Builtin<name, [Void, VectorType<Double, VSize>, Size, PointerType<Half, GenericAS>]>;
      }
    }
  }
}
// Variants using pointers to the constant address space.
foreach VSize = [2, 3, 4, 8, 16] in {
  foreach name = ["vload" # VSize] in {
    def : Builtin<name, [VectorType<Char, VSize>, Size, PointerType<ConstType<Char>, ConstantAS>]>;
    def : Builtin<name, [VectorType<UChar, VSize>, Size, PointerType<ConstType<UChar>, ConstantAS>]>;
    def : Builtin<name, [VectorType<Short, VSize>, Size, PointerType<ConstType<Short>, ConstantAS>]>;
    def : Builtin<name, [VectorType<UShort, VSize>, Size, PointerType<ConstType<UShort>, ConstantAS>]>;
    def : Builtin<name, [VectorType<Int, VSize>, Size, PointerType<ConstType<Int>, ConstantAS>]>;
    def : Builtin<name, [VectorType<UInt, VSize>, Size, PointerType<ConstType<UInt>, ConstantAS>]>;
    def : Builtin<name, [VectorType<Long, VSize>, Size, PointerType<ConstType<Long>, ConstantAS>]>;
    def : Builtin<name, [VectorType<ULong, VSize>, Size, PointerType<ConstType<ULong>, ConstantAS>]>;
    def : Builtin<name, [VectorType<Float, VSize>, Size, PointerType<ConstType<Float>, ConstantAS>]>;
    def : Builtin<name, [VectorType<Double, VSize>, Size, PointerType<ConstType<Double>, ConstantAS>]>;
    def : Builtin<name, [VectorType<Half, VSize>, Size, PointerType<ConstType<Half>, ConstantAS>]>;
  }
  foreach name = ["vloada_half" # VSize] in {
    def : Builtin<name, [VectorType<Float, VSize>, Size, PointerType<ConstType<Half>, ConstantAS>]>;
  }
  foreach rnd = ["", "_rte", "_rtz", "_rtp", "_rtn"] in {
    foreach name = ["vstorea_half" # VSize # rnd] in {
      def : Builtin<name, [Void, VectorType<Float, VSize>, Size, PointerType<Half, ConstantAS>]>;
      def : Builtin<name, [Void, VectorType<Double, VSize>, Size, PointerType<Half, ConstantAS>]>;
    }
  }
}
let MaxVersion = CL20 in {
  foreach AS = [GlobalAS, LocalAS, PrivateAS] in {
    def : Builtin<"vload_half", [Float, Size, PointerType<ConstType<Half>, AS>]>;
    foreach VSize = [2, 3, 4, 8, 16] in {
      foreach name = ["vload_half" # VSize] in {
        def : Builtin<name, [VectorType<Float, VSize>, Size, PointerType<ConstType<Half>, AS>]>;
      }
    }
    foreach rnd = ["", "_rte", "_rtz", "_rtp", "_rtn"] in {
      def : Builtin<"vstore_half" # rnd, [Void, Float, Size, PointerType<Half, AS>]>;
      def : Builtin<"vstore_half" # rnd, [Void, Double, Size, PointerType<Half, AS>]>;
      foreach VSize = [2, 3, 4, 8, 16] in {
        foreach name = ["vstore_half" # VSize # rnd] in {
          def : Builtin<name, [Void, VectorType<Float, VSize>, Size, PointerType<Half, AS>]>;
          def : Builtin<name, [Void, VectorType<Double, VSize>, Size, PointerType<Half, AS>]>;
        }
      }
    }
  }
}
let MinVersion = CL20 in {
  foreach AS = [GenericAS] in {
    def : Builtin<"vload_half", [Float, Size, PointerType<ConstType<Half>, AS>]>;
    foreach VSize = [2, 3, 4, 8, 16] in {
      foreach name = ["vload_half" # VSize] in {
        def : Builtin<name, [VectorType<Float, VSize>, Size, PointerType<ConstType<Half>, AS>]>;
      }
    }
    foreach rnd = ["", "_rte", "_rtz", "_rtp", "_rtn"] in {
      def : Builtin<"vstore_half" # rnd, [Void, Float, Size, PointerType<Half, AS>]>;
      def : Builtin<"vstore_half" # rnd, [Void, Double, Size, PointerType<Half, AS>]>;
      foreach VSize = [2, 3, 4, 8, 16] in {
        foreach name = ["vstore_half" # VSize # rnd] in {
          def : Builtin<name, [Void, VectorType<Float, VSize>, Size, PointerType<Half, AS>]>;
          def : Builtin<name, [Void, VectorType<Double, VSize>, Size, PointerType<Half, AS>]>;
        }
      }
    }
  }
}

foreach AS = [ConstantAS] in {
  def : Builtin<"vload_half", [Float, Size, PointerType<ConstType<Half>, AS>]>;
  foreach VSize = [2, 3, 4, 8, 16] in {
    foreach name = ["vload_half" # VSize] in {
      def : Builtin<name, [VectorType<Float, VSize>, Size, PointerType<ConstType<Half>, AS>]>;
    }
  }
}

//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.10, v1.2 s6.12.10, v2.0 s6.13.10: Async Copies from Global to Local Memory, Local to Global Memory, and Prefetch
// OpenCL Extension v2.0 s5.1.7 and s6.1.7: Async Copies from Global to Local Memory, Local to Global Memory, and Prefetch
// --- Table 18 ---
foreach name = ["async_work_group_copy"] in {
  def : Builtin<name, [Event, PointerType<AGenTypeN, LocalAS>, PointerType<ConstType<AGenTypeN>, GlobalAS>, Size, Event]>;
  def : Builtin<name, [Event, PointerType<AGenTypeN, GlobalAS>, PointerType<ConstType<AGenTypeN>, LocalAS>, Size, Event]>;
}
foreach name = ["async_work_group_strided_copy"] in {
  def : Builtin<name, [Event, PointerType<AGenTypeN, LocalAS>, PointerType<ConstType<AGenTypeN>, GlobalAS>, Size, Size, Event]>;
  def : Builtin<name, [Event, PointerType<AGenTypeN, GlobalAS>, PointerType<ConstType<AGenTypeN>, LocalAS>, Size, Size, Event]>;
}
foreach name = ["wait_group_events"] in {
  def : Builtin<name, [Void, Int, PointerType<Event, GenericAS>]>;
}
foreach name = ["prefetch"] in {
  def : Builtin<name, [Void, PointerType<ConstType<AGenTypeN>, GlobalAS>, Size]>;
}

//--------------------------------------------------------------------
// OpenCL v2.0 s6.13.11 - Atomics Functions.
// Functions that use memory_order and cl_mem_fence_flags enums are not
// declared here as the TableGen backend does not handle enums.

// OpenCL v1.0 s9.5, s9.6, s9.7 - Atomic Functions for 32-bit integers
// --- Table 9.1 ---
let Extension = FuncExtKhrGlobalInt32BaseAtomics in {
  foreach Type = [Int, UInt] in {
    foreach name = ["atom_add", "atom_sub", "atom_xchg"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, GlobalAS>, Type]>;
    }
    foreach name = ["atom_inc", "atom_dec"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, GlobalAS>]>;
    }
    foreach name = ["atom_cmpxchg"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, GlobalAS>, Type, Type]>;
    }
  }
}
// --- Table 9.3 ---
let Extension = FuncExtKhrLocalInt32BaseAtomics in {
  foreach Type = [Int, UInt] in {
    foreach name = ["atom_add", "atom_sub", "atom_xchg"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, LocalAS>, Type]>;
    }
    foreach name = ["atom_inc", "atom_dec"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, LocalAS>]>;
    }
    foreach name = ["atom_cmpxchg"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, LocalAS>, Type, Type]>;
    }
  }
}
// --- Table 9.5 ---
let Extension = FuncExtKhrInt64BaseAtomics in {
  foreach AS = [GlobalAS, LocalAS] in {
    foreach Type = [Long, ULong] in {
      foreach name = ["atom_add", "atom_sub", "atom_xchg"] in {
        def : Builtin<name, [Type, PointerType<VolatileType<Type>, AS>, Type]>;
      }
      foreach name = ["atom_inc", "atom_dec"] in {
        def : Builtin<name, [Type, PointerType<VolatileType<Type>, AS>]>;
      }
      foreach name = ["atom_cmpxchg"] in {
        def : Builtin<name, [Type, PointerType<VolatileType<Type>, AS>, Type, Type]>;
      }
    }
  }
}
// --- Table 9.2 ---
let Extension = FuncExtKhrGlobalInt32ExtendedAtomics in {
  foreach Type = [Int, UInt] in {
    foreach name = ["atom_min", "atom_max", "atom_and",
                    "atom_or", "atom_xor"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, GlobalAS>, Type]>;
    }
  }
}
// --- Table 9.4 ---
let Extension = FuncExtKhrLocalInt32ExtendedAtomics in {
  foreach Type = [Int, UInt] in {
    foreach name = ["atom_min", "atom_max", "atom_and",
                    "atom_or", "atom_xor"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, LocalAS>, Type]>;
    }
  }
}
// --- Table 9.6 ---
let Extension = FuncExtKhrInt64ExtendedAtomics in {
  foreach AS = [GlobalAS, LocalAS] in {
    foreach Type = [Long, ULong] in {
      foreach name = ["atom_min", "atom_max", "atom_and",
                      "atom_or", "atom_xor"] in {
        def : Builtin<name, [Type, PointerType<VolatileType<Type>, AS>, Type]>;
      }
    }
  }
}
// OpenCL v1.1 s6.11.1, v1.2 s6.12.11 - Atomic Functions
foreach AS = [GlobalAS, LocalAS] in {
  foreach Type = [Int, UInt] in {
    foreach name = ["atomic_add", "atomic_sub", "atomic_xchg",
                    "atomic_min", "atomic_max", "atomic_and",
                    "atomic_or", "atomic_xor"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, AS>, Type]>;
    }
    foreach name = ["atomic_inc", "atomic_dec"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, AS>]>;
    }
    foreach name = ["atomic_cmpxchg"] in {
      def : Builtin<name, [Type, PointerType<VolatileType<Type>, AS>, Type, Type]>;
    }
  }
}

//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.12, v1.2 s6.12.12, v2.0 s6.13.12 - Miscellaneous Vector Functions
// --- Table 19 ---
foreach VSize1 = [Vec2, Vec4, Vec8, Vec16] in {
  foreach VSize2 = [Vec2, Vec4, Vec8, Vec16] in {
    def : Builtin<"shuffle", [GenericType<"TLAll" # VSize1.Name, TLAll, VSize1>,
                              GenericType<"TLAll" # VSize2.Name, TLAll, VSize2>,
                              GenericType<"TLAllUnsigned" # VSize1.Name, TLAllUnsigned, VSize1>],
                  Attr.Const>;
  }
}
foreach VSize1 = [Vec2, Vec4, Vec8, Vec16] in {
  foreach VSize2 = [Vec2, Vec4, Vec8, Vec16] in {
    def : Builtin<"shuffle2", [GenericType<"TLAll" # VSize1.Name, TLAll, VSize1>,
                               GenericType<"TLAll" # VSize2.Name, TLAll, VSize2>,
                               GenericType<"TLAll" # VSize2.Name, TLAll, VSize2>,
                               GenericType<"TLAllUnsigned" # VSize1.Name, TLAllUnsigned, VSize1>],
                  Attr.Const>;
  }
}

//--------------------------------------------------------------------
// OpenCL v1.1 s6.11.3, v1.2 s6.12.14, v2.0 s6.13.14: Image Read and Write Functions
// OpenCL Extension v2.0 s5.1.8 and s6.1.8: Image Read and Write Functions
// --- Table 22: Image Read Functions with Samplers ---
foreach imgTy = [Image1d] in {
  foreach coordTy = [Int, Float] in {
    def : Builtin<"read_imagef", [VectorType<Float, 4>, ImageType<imgTy, "RO">, Sampler, coordTy], Attr.Pure>;
    def : Builtin<"read_imagei", [VectorType<Int, 4>, ImageType<imgTy, "RO">, Sampler, coordTy], Attr.Pure>;
    def : Builtin<"read_imageui", [VectorType<UInt, 4>, ImageType<imgTy, "RO">, Sampler, coordTy], Attr.Pure>;
  }
}
foreach imgTy = [Image2d, Image1dArray] in {
  foreach coordTy = [Int, Float] in {
    def : Builtin<"read_imagef", [VectorType<Float, 4>, ImageType<imgTy, "RO">, Sampler, VectorType<coordTy, 2>], Attr.Pure>;
    def : Builtin<"read_imagei", [VectorType<Int, 4>, ImageType<imgTy, "RO">, Sampler, VectorType<coordTy, 2>], Attr.Pure>;
    def : Builtin<"read_imageui", [VectorType<UInt, 4>, ImageType<imgTy, "RO">, Sampler, VectorType<coordTy, 2>], Attr.Pure>;
  }
}
foreach imgTy = [Image3d, Image2dArray] in {
  foreach coordTy = [Int, Float] in {
    def : Builtin<"read_imagef", [VectorType<Float, 4>, ImageType<imgTy, "RO">, Sampler, VectorType<coordTy, 4>], Attr.Pure>;
    def : Builtin<"read_imagei", [VectorType<Int, 4>, ImageType<imgTy, "RO">, Sampler, VectorType<coordTy, 4>], Attr.Pure>;
    def : Builtin<"read_imageui", [VectorType<UInt, 4>, ImageType<imgTy, "RO">, Sampler, VectorType<coordTy, 4>], Attr.Pure>;
  }
}
foreach coordTy = [Int, Float] in {
  def : Builtin<"read_imagef", [Float, ImageType<Image2dDepth, "RO">, Sampler, VectorType<coordTy, 2>], Attr.Pure>;
  def : Builtin<"read_imagef", [Float, ImageType<Image2dArrayDepth, "RO">, Sampler, VectorType<coordTy, 4>], Attr.Pure>;
}

// --- Table 23: Sampler-less Read Functions ---
foreach aQual = ["RO", "RW"] in {
  foreach imgTy = [Image2d, Image1dArray] in {
    def : Builtin<"read_imagef", [VectorType<Float, 4>, ImageType<imgTy, aQual>, VectorType<Int, 2>], Attr.Pure>;
    def : Builtin<"read_imagei", [VectorType<Int, 4>, ImageType<imgTy, aQual>, VectorType<Int, 2>], Attr.Pure>;
    def : Builtin<"read_imageui", [VectorType<UInt, 4>, ImageType<imgTy, aQual>, VectorType<Int, 2>], Attr.Pure>;
  }
  foreach imgTy = [Image3d, Image2dArray] in {
    def : Builtin<"read_imagef", [VectorType<Float, 4>, ImageType<imgTy, aQual>, VectorType<Int, 4>], Attr.Pure>;
    def : Builtin<"read_imagei", [VectorType<Int, 4>, ImageType<imgTy, aQual>, VectorType<Int, 4>], Attr.Pure>;
    def : Builtin<"read_imageui", [VectorType<UInt, 4>, ImageType<imgTy, aQual>, VectorType<Int, 4>], Attr.Pure>;
  }
  foreach imgTy = [Image1d, Image1dBuffer] in {
    def : Builtin<"read_imagef", [VectorType<Float, 4>, ImageType<imgTy, aQual>, Int], Attr.Pure>;
    def : Builtin<"read_imagei", [VectorType<Int, 4>, ImageType<imgTy, aQual>, Int], Attr.Pure>;
    def : Builtin<"read_imageui", [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Int], Attr.Pure>;
  }
  def : Builtin<"read_imagef", [Float, ImageType<Image2dDepth, aQual>, VectorType<Int, 2>], Attr.Pure>;
  def : Builtin<"read_imagef", [Float, ImageType<Image2dArrayDepth, aQual>, VectorType<Int, 4>], Attr.Pure>;
}

// --- Table 24: Image Write Functions ---
foreach aQual = ["WO", "RW"] in {
  foreach imgTy = [Image2d] in {
    def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, VectorType<Float, 4>]>;
    def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, VectorType<Int, 4>]>;
    def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, VectorType<UInt, 4>]>;
  }
  foreach imgTy = [Image2dArray] in {
    def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, VectorType<Float, 4>]>;
    def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, VectorType<Int, 4>]>;
    def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, VectorType<UInt, 4>]>;
  }
  foreach imgTy = [Image1d, Image1dBuffer] in {
    def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, Int, VectorType<Float, 4>]>;
    def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, Int, VectorType<Int, 4>]>;
    def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, Int, VectorType<UInt, 4>]>;
  }
  foreach imgTy = [Image1dArray] in {
    def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, VectorType<Float, 4>]>;
    def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, VectorType<Int, 4>]>;
    def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, VectorType<UInt, 4>]>;
  }
  foreach imgTy = [Image3d] in {
    def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, VectorType<Float, 4>]>;
    def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, VectorType<Int, 4>]>;
    def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, VectorType<UInt, 4>]>;
  }
  def : Builtin<"write_imagef", [Void, ImageType<Image2dDepth, aQual>, VectorType<Int, 2>, Float]>;
  def : Builtin<"write_imagef", [Void, ImageType<Image2dArrayDepth, aQual>, VectorType<Int, 4>, Float]>;
}

// --- Table 25: Image Query Functions ---
foreach aQual = ["RO", "WO", "RW"] in {
  foreach imgTy = [Image1d, Image1dBuffer, Image2d, Image3d,
                   Image1dArray, Image2dArray, Image2dDepth,
                   Image2dArrayDepth] in {
    foreach name = ["get_image_width", "get_image_channel_data_type",
                    "get_image_channel_order"] in {
      def : Builtin<name, [Int, ImageType<imgTy, aQual>]>;
    }
  }
  foreach imgTy = [Image2d, Image3d, Image2dArray, Image2dDepth,
                   Image2dArrayDepth] in {
    def : Builtin<"get_image_height", [Int, ImageType<imgTy, aQual>]>;
  }
  def : Builtin<"get_image_depth", [Int, ImageType<Image3d, aQual>]>;
  foreach imgTy = [Image2d, Image2dArray, Image2dDepth,
                   Image2dArrayDepth] in {
    def : Builtin<"get_image_dim", [VectorType<Int, 2>, ImageType<imgTy, aQual>]>;
  }
  def : Builtin<"get_image_dim", [VectorType<Int, 4>, ImageType<Image3d, aQual>]>;
  foreach imgTy = [Image1dArray, Image2dArray, Image2dArrayDepth] in {
    def : Builtin<"get_image_array_size", [Size, ImageType<imgTy, aQual>]>;
  }
}

// OpenCL extension v2.0 s5.1.9: Built-in Image Read Functions
// --- Table 8 ---
foreach aQual = ["RO"] in {
  foreach name = ["read_imageh"] in {
    foreach coordTy = [Int, Float] in {
      foreach imgTy = [Image2d, Image1dArray] in {
        def : Builtin<name, [VectorType<Half, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<coordTy, 2>], Attr.Pure>;
      }
      foreach imgTy = [Image3d, Image2dArray] in {
        def : Builtin<name, [VectorType<Half, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<coordTy, 4>], Attr.Pure>;
      }
      foreach imgTy = [Image1d] in {
        def : Builtin<name, [VectorType<Half, 4>, ImageType<imgTy, aQual>, Sampler, coordTy], Attr.Pure>;
      }
    }
  }
}
// OpenCL extension v2.0 s5.1.10: Built-in Image Sampler-less Read Functions
// --- Table 9 ---
foreach aQual = ["RO", "RW"] in {
  foreach name = ["read_imageh"] in {
    foreach imgTy = [Image2d, Image1dArray] in {
      def : Builtin<name, [VectorType<Half, 4>, ImageType<imgTy, aQual>, VectorType<Int, 2>], Attr.Pure>;
    }
    foreach imgTy = [Image3d, Image2dArray] in {
      def : Builtin<name, [VectorType<Half, 4>, ImageType<imgTy, aQual>, VectorType<Int, 4>], Attr.Pure>;
    }
    foreach imgTy = [Image1d, Image1dBuffer] in {
      def : Builtin<name, [VectorType<Half, 4>, ImageType<imgTy, aQual>, Int], Attr.Pure>;
    }
  }
}
// OpenCL extension v2.0 s5.1.11: Built-in Image Write Functions
// --- Table 10 ---
foreach aQual = ["WO", "RW"] in {
  foreach name = ["write_imageh"] in {
    def : Builtin<name, [Void, ImageType<Image2d, aQual>, VectorType<Int, 2>, VectorType<Half, 4>]>;
    def : Builtin<name, [Void, ImageType<Image2dArray, aQual>, VectorType<Int, 4>, VectorType<Half, 4>]>;
    def : Builtin<name, [Void, ImageType<Image1d, aQual>, Int, VectorType<Half, 4>]>;
    def : Builtin<name, [Void, ImageType<Image1dBuffer, aQual>, Int, VectorType<Half, 4>]>;
    def : Builtin<name, [Void, ImageType<Image1dArray, aQual>, VectorType<Int, 2>, VectorType<Half, 4>]>;
    def : Builtin<name, [Void, ImageType<Image3d, aQual>, VectorType<Int, 4>, VectorType<Half, 4>]>;
  }
}


//--------------------------------------------------------------------
// OpenCL v2.0 s6.13.15 - Work-group Functions
// --- Table 26 ---
let MinVersion = CL20 in {
  foreach name = ["work_group_all", "work_group_any"] in {
    def : Builtin<name, [Int, Int], Attr.Convergent>;
  }
  foreach name = ["work_group_broadcast"] in {
    def : Builtin<name, [IntLongFloatGenType1, IntLongFloatGenType1, Size], Attr.Convergent>;
    def : Builtin<name, [IntLongFloatGenType1, IntLongFloatGenType1, Size, Size], Attr.Convergent>;
    def : Builtin<name, [IntLongFloatGenType1, IntLongFloatGenType1, Size, Size, Size], Attr.Convergent>;
  }
  foreach op = ["add", "min", "max"] in {
    foreach name = ["work_group_reduce_", "work_group_scan_exclusive_",
                    "work_group_scan_inclusive_"] in {
      def : Builtin<name # op, [IntLongFloatGenType1, IntLongFloatGenType1], Attr.Convergent>;
    }
  }
}


// OpenCL v2.0 s9.17.3: Additions to section 6.13.1: Work-Item Functions
let MinVersion = CL20 in {
  let Extension = FuncExtKhrSubgroups in {
    def get_sub_group_size : Builtin<"get_sub_group_size", [UInt]>;
    def get_max_sub_group_size : Builtin<"get_max_sub_group_size", [UInt]>;
    def get_num_sub_groups : Builtin<"get_num_sub_groups", [UInt]>;
  }
}

//--------------------------------------------------------------------
// End of the builtin functions defined in the OpenCL C specification.
// Builtin functions defined in the OpenCL C Extension are below.
//--------------------------------------------------------------------


// OpenCL Extension v2.0 s9.18 - Mipmaps
let Extension = FuncExtKhrMipmapImage in {
  // Added to section 6.13.14.2.
  foreach aQual = ["RO"] in {
    foreach imgTy = [Image2d] in {
      foreach name = ["read_imagef"] in {
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, VectorType<Float, 2>, VectorType<Float, 2>], Attr.Pure>;
      }
      foreach name = ["read_imagei"] in {
        def : Builtin<name, [VectorType<Int, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<Int, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, VectorType<Float, 2>, VectorType<Float, 2>], Attr.Pure>;
      }
      foreach name = ["read_imageui"] in {
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, VectorType<Float, 2>, VectorType<Float, 2>], Attr.Pure>;
      }
    }
    foreach imgTy = [Image2dDepth] in {
      foreach name = ["read_imagef"] in {
        def : Builtin<name, [Float, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float], Attr.Pure>;
        def : Builtin<name, [Float, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, VectorType<Float, 2>, VectorType<Float, 2>], Attr.Pure>;
      }
    }
    foreach imgTy = [Image1d] in {
      foreach name = ["read_imagef"] in {
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, Float, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, Float, Float, Float], Attr.Pure>;
      }
      foreach name = ["read_imagei"] in {
        def : Builtin<name, [VectorType<Int, 4>, ImageType<imgTy, aQual>, Sampler, Float, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<Int, 4>, ImageType<imgTy, aQual>, Sampler, Float, Float, Float], Attr.Pure>;
      }
      foreach name = ["read_imageui"] in {
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, Float, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, Float, Float, Float], Attr.Pure>;
      }
    }
    foreach imgTy = [Image3d] in {
      foreach name = ["read_imagef"] in {
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, VectorType<Float, 4>, VectorType<Float, 4>], Attr.Pure>;
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, Float], Attr.Pure>;
      }
      foreach name = ["read_imagei"] in {
        def : Builtin<name, [VectorType<Int, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, VectorType<Float, 4>, VectorType<Float, 4>], Attr.Pure>;
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, Float], Attr.Pure>;
      }
      foreach name = ["read_imageui"] in {
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, VectorType<Float, 4>, VectorType<Float, 4>], Attr.Pure>;
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, Float], Attr.Pure>;
      }
    }
    foreach imgTy = [Image1dArray] in {
      foreach name = ["read_imagef"] in {
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float, Float], Attr.Pure>;
      }
      foreach name = ["read_imagei"] in {
        def : Builtin<name, [VectorType<Int, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<Int, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float, Float], Attr.Pure>;
      }
      foreach name = ["read_imageui"] in {
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 2>, Float, Float], Attr.Pure>;
      }
    }
    foreach imgTy = [Image2dArray] in {
      foreach name = ["read_imagef"] in {
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<Float, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, VectorType<Float, 2>, VectorType<Float, 2>], Attr.Pure>;
      }
      foreach name = ["read_imagei"] in {
        def : Builtin<name, [VectorType<Int, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<Int, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, VectorType<Float, 2>, VectorType<Float, 2>], Attr.Pure>;
      }
      foreach name = ["read_imageui"] in {
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, VectorType<Float, 2>, VectorType<Float, 2>], Attr.Pure>;
      }
    }
    foreach imgTy = [Image2dArrayDepth] in {
      foreach name = ["read_imagef"] in {
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, Float], Attr.Pure>;
        def : Builtin<name, [VectorType<UInt, 4>, ImageType<imgTy, aQual>, Sampler, VectorType<Float, 4>, VectorType<Float, 2>, VectorType<Float, 2>], Attr.Pure>;
      }
    }
  }
  // Added to section 6.13.14.4.
  foreach aQual = ["WO"] in {
    foreach imgTy = [Image2d] in {
      def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, Int, VectorType<Float, 4>]>;
      def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, Int, VectorType<Int, 4>]>;
      def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, Int, VectorType<UInt, 4>]>;
    }
    def : Builtin<"write_imagef", [Void, ImageType<Image2dDepth, aQual>, VectorType<Int, 2>, Int, Float]>;
    foreach imgTy = [Image1d] in {
      def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, Int, Int, VectorType<Float, 4>]>;
      def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, Int, Int, VectorType<Int, 4>]>;
      def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, Int, Int, VectorType<UInt, 4>]>;
    }
    foreach imgTy = [Image1dArray] in {
      def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, Int, VectorType<Float, 4>]>;
      def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, Int, VectorType<Int, 4>]>;
      def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, VectorType<Int, 2>, Int, VectorType<UInt, 4>]>;
    }
    foreach imgTy = [Image2dArray] in {
      def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, Int, VectorType<Float, 4>]>;
      def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, Int, VectorType<Int, 4>]>;
      def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, Int, VectorType<UInt, 4>]>;
    }
    def : Builtin<"write_imagef", [Void, ImageType<Image2dArrayDepth, aQual>, VectorType<Int, 4>, Int, Float]>;
    let Extension = FuncExtKhrMipmapAndWrite3d in {
      foreach imgTy = [Image3d] in {
        def : Builtin<"write_imagef", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, Int, VectorType<Float, 4>]>;
        def : Builtin<"write_imagei", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, Int, VectorType<Int, 4>]>;
        def : Builtin<"write_imageui", [Void, ImageType<imgTy, aQual>, VectorType<Int, 4>, Int, VectorType<UInt, 4>]>;
      }
    }
  }
  // Added to section 6.13.14.5
  foreach aQual = ["RO", "WO", "RW"] in {
    foreach imgTy = [Image1d, Image2d, Image3d, Image1dArray, Image2dArray, Image2dDepth, Image2dArrayDepth] in {
      def : Builtin<"get_image_num_mip_levels", [Int, ImageType<imgTy, aQual>]>;
    }
  }
}


//--------------------------------------------------------------------
// OpenCL Extension v2.0 s18.3 - Creating OpenCL Memory Objects from OpenGL MSAA Textures
let Extension = FuncExtKhrGlMsaaSharing in {
  // --- Table 6.13.14.3 ---
  foreach aQual = ["RO", "RW"] in {
    foreach imgTy = [Image2dMsaa] in {
      def : Builtin<"read_imagef", [VectorType<Float, 4>, ImageType<imgTy, aQual>, VectorType<Int, 2>, Int], Attr.Pure>;
      def : Builtin<"read_imagei", [VectorType<Int, 4>, ImageType<imgTy, aQual>, VectorType<Int, 2>, Int], Attr.Pure>;
      def : Builtin<"read_imageui", [VectorType<UInt, 4>, ImageType<imgTy, aQual>, VectorType<Int, 2>, Int], Attr.Pure>;
    }
    foreach imgTy = [Image2dArrayMsaa] in {
      def : Builtin<"read_imagef", [VectorType<Float, 4>, ImageType<imgTy, aQual>, VectorType<Int, 4>, Int], Attr.Pure>;
      def : Builtin<"read_imagei", [VectorType<Int, 4>, ImageType<imgTy, aQual>, VectorType<Int, 4>, Int], Attr.Pure>;
      def : Builtin<"read_imageui", [VectorType<UInt, 4>, ImageType<imgTy, aQual>, VectorType<Int, 4>, Int], Attr.Pure>;
    }
    foreach name = ["read_imagef"] in {
      def : Builtin<name, [Float, ImageType<Image2dMsaaDepth, aQual>, VectorType<Int, 2>, Int], Attr.Pure>;
      def : Builtin<name, [Float, ImageType<Image2dArrayMsaaDepth, aQual>, VectorType<Int, 4>, Int], Attr.Pure>;
    }
  }

  // --- Table 6.13.14.5 ---
  foreach aQual = ["RO", "WO", "RW"] in {
    foreach imgTy = [Image2dMsaa, Image2dArrayMsaa, Image2dMsaaDepth, Image2dArrayMsaaDepth] in {
      foreach name = ["get_image_width", "get_image_height",
                      "get_image_channel_data_type", "get_image_channel_order",
                      "get_image_num_samples"] in {
        def : Builtin<name, [Int, ImageType<imgTy, aQual>], Attr.Const>;
      }
      def : Builtin<"get_image_dim", [VectorType<Int, 2>, ImageType<imgTy, aQual>], Attr.Const>;
    }
    def : Builtin<"get_image_array_size", [Size, ImageType<Image2dArrayMsaaDepth, aQual>], Attr.Const>;
  }
}
