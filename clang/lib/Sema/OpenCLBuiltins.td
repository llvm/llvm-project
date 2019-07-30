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


// Qualified Type. Allow to retrieve one ASTContext QualType.
class QualType<string _Name> {
  // Name of the field or function in a clang::ASTContext
  // E.g. Name="IntTy" for the int type, and "getIntPtrType()" for an intptr_t
  string Name = _Name;
}

// Helper class to store type access qualifiers (volatile, const, ...).
class Qualifier<string _QualName> {
  string QualName = _QualName;
}

//===----------------------------------------------------------------------===//
//                      OpenCL C classes for types
//===----------------------------------------------------------------------===//
// OpenCL types (int, float, ...)
class Type<string _Name, QualType _QTName> {
  // Name of the Type
  string Name = _Name;
  // QualType associated with this type
  QualType QTName = _QTName;
  // Size of the vector (if applicable)
  int VecWidth = 0;
  // Is pointer
  bit IsPointer = 0;
  // List of qualifiers associated with the type (volatile, ...)
  list<Qualifier> QualList = [];
  // Address space
  string AddrSpace = "clang::LangAS::Default";
  // Access qualifier. Must be one of ("RO", "WO", "RW").
  string AccessQualifier = "";
}

// OpenCL vector types (e.g. int2, int3, int16, float8, ...)
class VectorType<Type _Ty, int _VecWidth> : Type<_Ty.Name, _Ty.QTName> {
  int VecWidth = _VecWidth;
}

// OpenCL pointer types (e.g. int*, float*, ...)
class PointerType<Type _Ty, AddressSpace _AS = GlobalAS> :
                                      Type<_Ty.Name, _Ty.QTName> {
  bit IsPointer = 1;
  string AddrSpace = _AS.Name;
}

// OpenCL image types (e.g. image2d_t, ...)
class ImageType<Type _Ty, QualType _QTName, string _AccessQualifier> :
                                              Type<_Ty.Name, _QTName> {
  let AccessQualifier = _AccessQualifier;
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
//                           Multiclass definitions
//===----------------------------------------------------------------------===//
// multiclass BifN: Creates Builtin class instances for OpenCL builtin
//                  functions with N arguments.
// _Name      : Name of the function
// _Signature : Signature of the function (list of the Type used by the
//              function, the first one being the return type).
// _IsVector  : List of bit indicating if the type in the _Signature at the
//              same index is to be a vector in the multiple overloads. The
//              list must have at least one non-zero value.
multiclass Bif0<string _Name, list<Type> _Signature, list<bit> _IsVector> {
  def : Builtin<_Name, _Signature>;
  foreach v = [2, 3, 4, 8, 16] in {
    def : Builtin<_Name,
                  [!if(_IsVector[0], VectorType<_Signature[0], v>, _Signature[0])]>;
  }
}
multiclass Bif1<string _Name, list<Type> _Signature, list<bit> _IsVector> {
  def : Builtin<_Name, _Signature>;
  foreach v = [2, 3, 4, 8, 16] in {
    def : Builtin<_Name,
                  [!if(_IsVector[0], VectorType<_Signature[0], v>, _Signature[0]),
                  !if(_IsVector[1], VectorType<_Signature[1], v>, _Signature[1])]>;
  }
}
multiclass Bif2<string _Name, list<Type> _Signature, list<bit> _IsVector> {
  def : Builtin<_Name, _Signature>;
  foreach v = [2, 3, 4, 8, 16] in {
    def : Builtin<_Name,
                  [!if(_IsVector[0], VectorType<_Signature[0], v>, _Signature[0]),
                  !if(_IsVector[1], VectorType<_Signature[1], v>, _Signature[1]),
                  !if(_IsVector[2], VectorType<_Signature[2], v>, _Signature[2])]>;
  }
}
multiclass Bif3<string _Name, list<Type> _Signature, list<bit> _IsVector> {
  def : Builtin<_Name, _Signature>;
  foreach v = [2, 3, 4, 8, 16] in {
    def : Builtin<_Name,
                  [!if(_IsVector[0], VectorType<_Signature[0], v>, _Signature[0]),
                  !if(_IsVector[1], VectorType<_Signature[1], v>, _Signature[1]),
                  !if(_IsVector[2], VectorType<_Signature[2], v>, _Signature[2]),
                  !if(_IsVector[3], VectorType<_Signature[3], v>, _Signature[3])]>;
  }
}
//===----------------------------------------------------------------------===//
//                 Definitions of OpenCL C types
//===----------------------------------------------------------------------===//
// OpenCL v1.2 s6.1.1: Built-in Scalar Data Types
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

// OpenCL v1.2 s6.1.2: Built-in Vector Data Types
foreach v = [2, 3, 4, 8, 16] in {
  def char#v#_t    : VectorType<Char, v>;
  def uchar#v#_t   : VectorType<UChar, v>;
  def short#v#_t   : VectorType<Short, v>;
  def ushort#v#_t  : VectorType<UShort, v>;
  def "int"#v#_t   : VectorType<Int, v>;
  def uint#v#_t    : VectorType<UInt, v>;
  def long#v#_t    : VectorType<Long, v>;
  def ulong#v#_t   : VectorType<ULong, v>;
  def float#v#_t   : VectorType<Float, v>;
  def double#v#_t  : VectorType<Double, v>;
  def half#v#_t    : VectorType<Half, v>;
}

// OpenCL v1.2 s6.1.3: Other Built-in Data Types
// These definitions with a "null" name are "abstract". They should not
// be used in definitions of Builtin functions.
def image2d_t         : Type<"image2d_t", QualType<"null">>;
def image3d_t         : Type<"image3d_t", QualType<"null">>;
def image2d_array_t   : Type<"image2d_array_t", QualType<"null">>;
def image1d_t         : Type<"image1d_t", QualType<"null">>;
def image1d_buffer_t  : Type<"image1d_buffer_t", QualType<"null">>;
def image1d_array_t   : Type<"image1d_array_t", QualType<"null">>;
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
  foreach type = [Float, Double, Half] in {
    defm : Bif1<name, [type, type], [1, 1]>;
  }
}

foreach name = ["atan2", "atan2pi"] in {
  foreach type = [Float, Double, Half] in {
    defm : Bif2<name, [type, type, type], [1, 1, 1]>;
  }
}

foreach name = ["fmax", "fmin"] in {
  foreach type = [Float, Double, Half] in {
    defm : Bif2<name, [type, type, type], [1, 1, 1]>;
    defm : Bif2<name, [type, type, type], [1, 1, 0]>;
  }
}

// OpenCL v1.2 s6.12.14: Built-in Image Read Functions
def read_imagef : Builtin<"read_imagef",
                          [float4_t, image2d_RO_t, VectorType<Int, 2>]>;
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
