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
  int Version = _Version;
}
def CL10: Version<100>;
def CL11: Version<110>;
def CL12: Version<120>;
def CL20: Version<200>;

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
// Its Child classes can represent concrete types (e.g.: VectorType) or
// custom types (e.g.: GenType).
// Instances of these child classes should be used in Builtin function
// arguments.  See the definition of the "read_imagef" function as example.
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
  // Inherited fields
  let IsPointer = _Ty.IsPointer;
  let IsConst = _Ty.IsConst;
  let IsVolatile = _Ty.IsVolatile;
  let AccessQualifier = _Ty.AccessQualifier;
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

// OpenCL image types (e.g. image2d_t, ...)
class ImageType<Type _Ty, QualType _QTName, string _AccessQualifier> :
                                              Type<_Ty.Name, _QTName> {
  let AccessQualifier = _AccessQualifier;
}

// List of Types.
class TypeList<string _Name, list<Type> _Type> {
  string Name = _Name;
  list<Type> List = _Type;
}

// A GenericType is an abstract type that defines a set of types as a
// combination of Types and vector sizes.
//
// E.g.: If TypeList = <int, float> and VectorList = <1, 2, 4>, then it
//       represents <int, int2, int4, float, float2, float4>.
// _Ty          : Name of the GenType.
// _TypeList    : List of basic data Types.
// _VectorList  : Sizes of the vector for each type of the _TypeList, 1 being a
//                scalar.
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

//===----------------------------------------------------------------------===//
//                      OpenCL C class for builtin functions
//===----------------------------------------------------------------------===//
class Builtin<string _Name, list<Type> _Signature> {
  // Name of the builtin function
  string Name = _Name;
  // List of types used by the function. The first one is the return type and
  // the following are the arguments. The list must have at least one element
  // (the return type).
  list<Type> Signature = _Signature;
  // OpenCL Extension to which the function belongs (cl_khr_subgroups, ...)
  string Extension = "";
  // OpenCL Version to which the function belongs (CL10, ...)
  Version Version = CL10;
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

// OpenCL v1.2 s6.1.3: Other Built-in Data Types
// These definitions with a "null" name are "abstract". They should not
// be used in definitions of Builtin functions.
def image2d_t         : Type<"image2d_t", QualType<"null", 1>>;
def image3d_t         : Type<"image3d_t", QualType<"null", 1>>;
def image2d_array_t   : Type<"image2d_array_t", QualType<"null", 1>>;
def image1d_t         : Type<"image1d_t", QualType<"null", 1>>;
def image1d_buffer_t  : Type<"image1d_buffer_t", QualType<"null", 1>>;
def image1d_array_t   : Type<"image1d_array_t", QualType<"null", 1>>;
// Unlike the few functions above, the following definitions can be used
// in definitions of Builtin functions (they have a QualType with a name).
foreach v = ["RO", "WO", "RW"] in {
  def image2d_#v#_t       : ImageType<image2d_t,
                                      QualType<"OCLImage2d"#v#"Ty">,
                                      v>;
  def image3d_#v#_t       : ImageType<image3d_t,
                                      QualType<"OCLImage3d"#v#"Ty">,
                                      v>;
  def image2d_array#v#_t  : ImageType<image2d_array_t,
                                      QualType<"OCLImage2dArray"#v#"Ty">,
                                      v>;
  def image1d_#v#_t       : ImageType<image1d_t,
                                      QualType<"OCLImage1d"#v#"Ty">,
                                      v>;
  def image1d_buffer#v#_t : ImageType<image1d_buffer_t,
                                      QualType<"OCLImage1dBuffer"#v#"Ty">,
                                      v>;
  def image1d_array#v#_t  : ImageType<image1d_array_t,
                                      QualType<"OCLImage1dArray"#v#"Ty">,
                                      v>;
}

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

// Type lists.
def TLAll   : TypeList<"TLAll", [Char, UChar, Short, UShort, Int, UInt, Long, ULong, Float, Double, Half]>;
def TLFloat : TypeList<"TLFloat", [Float, Double, Half]>;

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
// Float
def FGenTypeN              : GenericType<"FGenTypeN", TLFloat, VecAndScalar>;

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
        def : Builtin<"convert_" # RType.Name # sat # rnd, [RType, IType]>;
        foreach v = [2, 3, 4, 8, 16] in {
          def : Builtin<"convert_" # RType.Name # v # sat # rnd,
                        [VectorType<RType, v>,
                         VectorType<IType, v>]>;
        }
      }
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

// OpenCL v1.0 s9.5, s9.6, s9.7 - Atomic Functions for 32-bit integers.
// --- Table 9.1 ---
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

// OpenCL v1.2 s6.12.1: Work-Item Functions
def get_work_dim : Builtin<"get_work_dim", [UInt]>;
foreach name = ["get_global_size", "get_global_id", "get_local_size",
                "get_local_id", "get_num_groups", "get_group_id",
                "get_global_offset"] in {
  def : Builtin<name, [Size, UInt]>;
}

// OpenCL v1.2 s6.12.2: Math Functions
foreach name = ["acos", "acosh", "acospi",
                "asin", "asinh", "asinpi",
                "atan", "atanh", "atanpi"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN]>;
}

foreach name = ["atan2", "atan2pi"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN, FGenTypeN]>;
}

foreach name = ["fmax", "fmin"] in {
  def : Builtin<name, [FGenTypeN, FGenTypeN, FGenTypeN]>;
  def : Builtin<name, [GenTypeFloatVecNoScalar, GenTypeFloatVecNoScalar, Float]>;
  def : Builtin<name, [GenTypeDoubleVecNoScalar, GenTypeDoubleVecNoScalar, Double]>;
  def : Builtin<name, [GenTypeHalfVecNoScalar, GenTypeHalfVecNoScalar, Half]>;
}

// OpenCL v1.1 s6.11.3, v1.2 s6.12.3, v2.0 s6.13.3 - Integer Functions
foreach name = ["max", "min"] in {
  def : Builtin<name, [AIGenTypeN, AIGenTypeN, AIGenTypeN]>;
  def : Builtin<name, [AIGenTypeNNoScalar, AIGenTypeNNoScalar, AIGenType1]>;
}

// OpenCL v1.2 s6.12.14: Built-in Image Read Functions
def read_imagef : Builtin<"read_imagef",
                          [VectorType<Float, 4>, image2d_RO_t, VectorType<Int, 2>]>;
def write_imagef : Builtin<"write_imagef",
                           [Void,
                            image2d_WO_t,
                            VectorType<Int, 2>,
                            VectorType<Float, 4>]>;


// OpenCL v2.0 s9.17.3: Additions to section 6.13.1: Work-Item Functions
let Version = CL20 in {
  let Extension = "cl_khr_subgroups" in {
    def get_sub_group_size : Builtin<"get_sub_group_size", [UInt]>;
    def get_max_sub_group_size : Builtin<"get_max_sub_group_size", [UInt]>;
    def get_num_sub_groups : Builtin<"get_num_sub_groups", [UInt]>;
  }
}
