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
  string AddrSpace = _AS;
}
def default_as    : AddressSpace<"clang::LangAS::Default">;
def private_as    : AddressSpace<"clang::LangAS::opencl_private">;
def global_as     : AddressSpace<"clang::LangAS::opencl_global">;
def constant_as   : AddressSpace<"clang::LangAS::opencl_constant">;
def local_as      : AddressSpace<"clang::LangAS::opencl_local">;
def generic_as    : AddressSpace<"clang::LangAS::opencl_generic">;


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
class PointerType<Type _Ty, AddressSpace _AS = global_as> :
                                      Type<_Ty.Name, _Ty.QTName> {
  bit IsPointer = 1;
  string AddrSpace = _AS.AddrSpace;
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
def bool_t      : Type<"bool", QualType<"BoolTy">>;
def char_t      : Type<"char", QualType<"CharTy">>;
def uchar_t     : Type<"uchar", QualType<"UnsignedCharTy">>;
def short_t     : Type<"short", QualType<"ShortTy">>;
def ushort_t    : Type<"ushort", QualType<"UnsignedShortTy">>;
def int_t       : Type<"int", QualType<"IntTy">>;
def uint_t      : Type<"uint", QualType<"UnsignedIntTy">>;
def long_t      : Type<"long", QualType<"LongTy">>;
def ulong_t     : Type<"ulong", QualType<"UnsignedLongTy">>;
def float_t     : Type<"float", QualType<"FloatTy">>;
def double_t    : Type<"double", QualType<"DoubleTy">>;
def half_t      : Type<"half", QualType<"HalfTy">>;
def size_t      : Type<"size_t",  QualType<"getSizeType()">>;
def ptrdiff_t   : Type<"ptrdiff_t", QualType<"getPointerDiffType()">>;
def intptr_t    : Type<"intptr_t", QualType<"getIntPtrType()">>;
def uintptr_t   : Type<"uintptr_t", QualType<"getUIntPtrType()">>;
def void_t      : Type<"void", QualType<"VoidTy">>;

// OpenCL v1.2 s6.1.2: Built-in Vector Data Types
foreach v = [2, 3, 4, 8, 16] in {
  def char#v#_t    : VectorType<char_t, v>;
  def uchar#v#_t   : VectorType<uchar_t, v>;
  def short#v#_t   : VectorType<short_t, v>;
  def ushort#v#_t  : VectorType<ushort_t, v>;
  def "int"#v#_t   : VectorType<int_t, v>;
  def uint#v#_t    : VectorType<uint_t, v>;
  def long#v#_t    : VectorType<long_t, v>;
  def ulong#v#_t   : VectorType<ulong_t, v>;
  def float#v#_t   : VectorType<float_t, v>;
  def double#v#_t  : VectorType<double_t, v>;
  def half#v#_t    : VectorType<half_t, v>;
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

def sampler_t         : Type<"sampler_t", QualType<"OCLSamplerTy">>;
def event_t           : Type<"event_t", QualType<"OCLEventTy">>;

//===----------------------------------------------------------------------===//
//                 Definitions of OpenCL builtin functions
//===----------------------------------------------------------------------===//
// OpenCL v1.2 s6.2.3: Explicit Conversions
// Generate the convert_ builtins.
foreach RType = [float_t, double_t, char_t, uchar_t, short_t, ushort_t,
                int_t, uint_t, long_t, ulong_t] in {
  foreach IType = [float_t, double_t, char_t, uchar_t, short_t, ushort_t,
                   int_t, uint_t, long_t, ulong_t] in {
    foreach sat = ["", "_sat"] in {
      foreach rte = ["", "_rte", "_rtz", "_rtp", "_rtn"] in {
        def : Builtin<"convert_"  # RType.Name # sat # rte, [RType, IType]>;
        foreach v = [2, 3, 4, 8, 16] in {
          def : Builtin<"convert_" # RType.Name # v # sat # rte,
                        [VectorType<RType, v>,
                         VectorType<IType, v>]>;
        }
      }
    }
  }
}

// OpenCL v1.2 s6.12.1: Work-Item Functions
def get_work_dim : Builtin<"get_work_dim", [uint_t]>;
foreach name = ["get_global_size", "get_global_id", "get_local_size",
                "get_local_id", "get_num_groups", "get_group_id",
                "get_global_offset"] in {
  def : Builtin<name, [size_t, uint_t]>;
}

// OpenCL v1.2 s6.12.2: Math Functions
foreach name = ["acos", "acosh", "acospi",
                "asin", "asinh", "asinpi",
                "atan", "atanh", "atanpi"] in {
  foreach type = [float_t, double_t, half_t] in {
    defm : Bif1<name, [type, type], [1, 1]>;
  }
}

foreach name = ["atan2", "atan2pi"] in {
  foreach type = [float_t, double_t, half_t] in {
    defm : Bif2<name, [type, type, type], [1, 1, 1]>;
  }
}

foreach name = ["fmax", "fmin"] in {
  foreach type = [float_t, double_t, half_t] in {
    defm : Bif2<name, [type, type, type], [1, 1, 1]>;
    defm : Bif2<name, [type, type, type], [1, 1, 0]>;
  }
}

// OpenCL v1.2 s6.12.14: Built-in Image Read Functions
def read_imagef : Builtin<"read_imagef",
                          [float4_t, image2d_RO_t, VectorType<int_t, 2>]>;
def write_imagef : Builtin<"write_imagef",
                           [void_t,
                            image2d_WO_t,
                            VectorType<int_t, 2>,
                            VectorType<float_t, 4>]>;


// OpenCL v2.0 s9.17.3: Additions to section 6.13.1: Work-Item Functions
let Version = CL20 in {
  let Extension = "cl_khr_subgroups" in {
    def get_sub_group_size : Builtin<"get_sub_group_size", [uint_t]>;
    def get_max_sub_group_size : Builtin<"get_max_sub_group_size", [uint_t]>;
    def get_num_sub_groups : Builtin<"get_num_sub_groups", [uint_t]>;
  }
}
