//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines HLSL intrinsic functions in tablegen. The HLSLEmitter
// backend processes these definitions and generates two .inc files:
//   - hlsl_alias_intrinsics_gen.inc: builtin alias declarations using
//     _HLSL_BUILTIN_ALIAS, included by hlsl_alias_intrinsics.h.
//   - hlsl_inline_intrinsics_gen.inc: inline function definitions (detail
//     helper calls and literal bodies), included by hlsl_intrinsics.h.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Argument and return type base classes
//===----------------------------------------------------------------------===//

// Base class for argument and return type positions.
class HLSLArgType;

// Base class for return type positions.
class HLSLReturnType;

// Void return type.
def VoidTy : HLSLReturnType;

//===----------------------------------------------------------------------===//
// HLSL element types
//===----------------------------------------------------------------------===//

// Represents a concrete HLSL scalar element type.
// Can be used directly as an argument or return type for a fixed scalar
// (e.g., FloatTy in Args produces a 'float' argument).
class HLSLType<string name> : HLSLArgType, HLSLReturnType {
  string Name = name;
  string TypeName = name;

  // When set, overloads using this type are guarded by
  // #ifdef __HLSL_ENABLE_16_BIT and emitted with
  // _HLSL_AVAILABILITY(shadermodel, 6.2), or the intrinsic's Availability
  // if it is greater.
  bit Is16Bit = 0;

  // When set, overloads using this type are emitted with
  // _HLSL_16BIT_AVAILABILITY(shadermodel, 6.2) instead of _HLSL_AVAILABILITY.
  // This macro expands to an availability attribute only when
  // __HLSL_ENABLE_16_BIT is defined (i.e. half is a true 16-bit float);
  // otherwise it expands to nothing since half is an alias for float.
  // If the intrinsic's Availability is >= SM6.2, _HLSL_AVAILABILITY is used
  // instead because 16-bit support is already implied.
  bit IsConditionally16Bit = 0;
}

def BoolTy   : HLSLType<"bool">;
def HalfTy   : HLSLType<"half">   { let IsConditionally16Bit = 1; }
def FloatTy  : HLSLType<"float">;
def DoubleTy : HLSLType<"double">;
def Int16Ty  : HLSLType<"int16_t">  { let Is16Bit = 1; }
def UInt16Ty : HLSLType<"uint16_t"> { let Is16Bit = 1; }
def IntTy    : HLSLType<"int">;
def UIntTy   : HLSLType<"uint">;
def Int64Ty  : HLSLType<"int64_t">;
def UInt64Ty : HLSLType<"uint64_t">;

//===----------------------------------------------------------------------===//
// Element type groups
//===----------------------------------------------------------------------===//

defvar AllFloatTypes = [HalfTy, FloatTy, DoubleTy];
defvar SignedIntTypes = [Int16Ty, IntTy, Int64Ty];
defvar UnsignedIntTypes = [UInt16Ty, UIntTy, UInt64Ty];
defvar AllIntTypes = [Int16Ty, UInt16Ty, IntTy, UIntTy,
                      Int64Ty, UInt64Ty];
defvar SignedTypes = [Int16Ty, HalfTy, IntTy, FloatTy,
                      Int64Ty, DoubleTy];
defvar AllNumericTypes = [Int16Ty, UInt16Ty, HalfTy, IntTy, UIntTy,
                          FloatTy, Int64Ty, UInt64Ty, DoubleTy];
defvar AllTypesWithBool = [BoolTy, Int16Ty, UInt16Ty, HalfTy,
                           IntTy, UIntTy, FloatTy, Int64Ty,
                           UInt64Ty, DoubleTy];
defvar NumericTypesNoDbl = [Int16Ty, UInt16Ty, HalfTy, IntTy, UIntTy,
                            FloatTy, Int64Ty, UInt64Ty];

//===----------------------------------------------------------------------===//
// Argument/return types
//
// These classes are usable in both argument and return type positions.
//===----------------------------------------------------------------------===//

// The varying type — expanded per VaryingTypes.
// As an argument: the arg type varies with each overload.
// As a return type: returns the same type as the varying arg.
def Varying : HLSLArgType, HLSLReturnType;

// The scalar element of the varying type.
// As an argument: always the scalar element type regardless of overload shape.
// As a return type: returns the scalar element type (e.g., float dot(float3, float3)).
def VaryingElemType : HLSLArgType, HLSLReturnType;

// The varying shape with a fixed element type.
// As an argument: same shape as Varying but with the given element type.
// As a return type: same shape as the varying arg but with the given element type.
// For example, VaryingShape<UIntTy> with a float3 overload produces uint3.
class VaryingShape<HLSLType ty> : HLSLArgType, HLSLReturnType {
  HLSLType ElementType = ty;
}

// A concrete vector type (e.g., VectorType<UIntTy, 4> -> uint4).
// As an argument: the arg is always this vector type.
// As a return type: always returns this vector type, ignoring argument shape.
class VectorType<HLSLType ty, int size> : HLSLArgType, HLSLReturnType {
  HLSLType ElementType = ty;
  int Size = size;
}

//===----------------------------------------------------------------------===//
// Shader model versions
//===----------------------------------------------------------------------===//

// Represents a shader model version
class ShaderModel<int major, int minor> {
  int Major = major;
  int Minor = minor;
}

// Sentinel: no shader model requirement.
def NoSM  : ShaderModel<0, 0>;

// Valid Shader Model records
foreach i = 0...9 in {
  def SM6_ #i : ShaderModel<6, i>;
}

//===----------------------------------------------------------------------===//
// Matrix dimension records
//===----------------------------------------------------------------------===//

class MatDim<int rows, int cols> {
  int Rows = rows;
  int Cols = cols;
}

foreach r = 1...4 in
  foreach c = 1...4 in
    def Mat#r#"x"#c : MatDim<r, c>;

// All non-1x1 matrix dimensions (1x2 through 4x4).
defvar AllMatDims = [Mat1x2, Mat1x3, Mat1x4,
                     Mat2x1, Mat2x2, Mat2x3, Mat2x4,
                     Mat3x1, Mat3x2, Mat3x3, Mat3x4,
                     Mat4x1, Mat4x2, Mat4x3, Mat4x4];

//===----------------------------------------------------------------------===//
// HLSLBuiltin class
//===----------------------------------------------------------------------===//

class HLSLBuiltin<string name, string builtin = ""> {
  string Name = name;

  // When set, generates a _HLSL_BUILTIN_ALIAS(Builtin) declaration that
  // aliases the named Clang builtin. Mutually exclusive with DetailFunc
  // and Body.
  string Builtin = builtin;

  // Doxygen documentation comment emitted before overloads in generated code.
  string Doc = "";

  // When set, generates an inline function body calling
  // __detail::DetailFunc(args...) instead of _HLSL_BUILTIN_ALIAS(Builtin).
  // Parameters are named p0, p1, p2, ... by default, or use ParamNames to
  // specify custom names. Mutually exclusive with Body and Builtin.
  string DetailFunc = "";

  // When set, generates an inline function with this literal body text.
  // Intended for single-statement functions. Multi-line functions should
  // instead be defined as a helper and called with DetailFunc.
  // Parameters are named p0, p1, p2, ... by default, or use ParamNames to
  // specify custom names. Mutually exclusive with DetailFunc and Builtin.
  code Body = "";

  // Determines how the return type is derived for each overload.
  HLSLReturnType ReturnType = VoidTy;

  // Argument list. Each entry is either:
  //   Varying     - type varies with VaryingTypes (expanded per type)
  //   HLSLType    - a fixed scalar type at that position (e.g., UIntTy)
  //   VectorType  - a fixed vector type at that position
  // The number of arguments is deduced from the length of this list.
  // Examples:
  //   [Varying]                              -> func(T)
  //   [Varying, Varying, Varying]            -> func(T, T, T)
  //   [Varying, UIntTy]                      -> func(T, uint)
  //   [UIntTy, UIntTy, IntTy]                -> func(uint, uint, int)
  //   []                                     -> func()
  list<HLSLArgType> Args = [];

  // Custom parameter names for generated functions.
  // When empty, inline functions (Body or DetailFunc) use p0, p1, p2, ...
  // and alias functions omit parameter names.
  list<string> ParamNames = [];

  // When set, emits 'constexpr' instead of 'inline' for inline functions
  // (i.e., functions using Body or DetailFunc).
  bit IsConstexpr = 0;

  // Whether the function has the convergent attribute
  bit IsConvergent = 0;

  // Argument element types — drives overload expansion.
  // One overload set is generated per type (scalar + vectors + matrices).
  // Only used when Args contains Varying entries.
  list<HLSLType> VaryingTypes = [];

  // Whether to generate scalar overloads for Varying typed arguments.
  bit VaryingScalar = 0;

  // Vector sizes to generate for Varying typed arguments (e.g., [2,3,4]).
  list<int> VaryingVecSizes = [];

  // Matrix dimensions to generate for Varying typed arguments.
  list<MatDim> VaryingMatDims = [];

  // Default shader model availability version for all types.
  // Use NoSM for no availability requirement.
  ShaderModel Availability = NoSM;
}

//===----------------------------------------------------------------------===//
// HLSLBuiltin helper subclasses
//===----------------------------------------------------------------------===//

// T func(T) with scalar + vec2/3/4 + matrix overloads.
class HLSLOneArgBuiltin<string name, string builtin>
    : HLSLBuiltin<name, builtin> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = AllMatDims;
}

// T func(T, T) with scalar + vec2/3/4 + matrix overloads.
class HLSLTwoArgBuiltin<string name, string builtin>
    : HLSLBuiltin<name, builtin> {
  let Args = [Varying, Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = AllMatDims;
}

// T func(T, T, T) with scalar + vec2/3/4 + matrix overloads.
class HLSLThreeArgBuiltin<string name, string builtin>
    : HLSLBuiltin<name, builtin> {
  let Args = [Varying, Varying, Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = AllMatDims;
}

// Detail function base: generates inline function bodies calling
// __detail::DetailFunc(args...) instead of _HLSL_BUILTIN_ALIAS.
class HLSLDetail<string name, string detail> : HLSLBuiltin<name> {
  let DetailFunc = detail;
}

// T func(T) with scalar + vec2/3/4 + matrix overloads.
class HLSLOneArgDetail<string name, string detail>
    : HLSLDetail<name, detail> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = AllMatDims;
}

// T func(T, T) with scalar + vec2/3/4 + matrix overloads.
class HLSLTwoArgDetail<string name, string detail>
    : HLSLDetail<name, detail> {
  let Args = [Varying, Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = AllMatDims;
}

// T func(T, T, T) with scalar + vec2/3/4 + matrix overloads.
class HLSLThreeArgDetail<string name, string detail>
    : HLSLDetail<name, detail> {
  let Args = [Varying, Varying, Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = AllMatDims;
}

// Inline body variant: T func(T) with a literal inline body (no builtin alias).
// Body must be specified (e.g., let Body = "return p0;").
class HLSLOneArgInlineBuiltin<string name> : HLSLBuiltin<name> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = AllMatDims;
}

//===----------------------------------------------------------------------===//
// Intrinsic definitions (sorted alphabetically by function name)
//===----------------------------------------------------------------------===//

// Returns the absolute value of the input value, Val.
def hlsl_abs : HLSLOneArgBuiltin<"abs", "__builtin_elementwise_abs"> {
  let Doc = [{
\fn T abs(T Val)
\brief Returns the absolute value of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = SignedTypes;
  let VaryingMatDims = [];
}

// Unsigned abs is a constexpr identity — unsigned values are already non-negative.
def hlsl_abs_unsigned : HLSLOneArgInlineBuiltin<"abs"> {
  let ParamNames = ["V"];
  let Body = "return V;";
  let IsConstexpr = 1;
  let VaryingTypes = UnsignedIntTypes;
  let VaryingMatDims = [];
}

// Returns the boolean AND of two bool values or vectors.
def hlsl_and : HLSLTwoArgBuiltin<"and", "__builtin_hlsl_and"> {
  let Doc = [{
\fn bool and(bool x, bool y)
\brief Logically ands two boolean vectors or matrices elementwise and
produces a bool vector or matrix output.
}];
  let VaryingTypes = [BoolTy];
}

// Returns the cross product of two floating-point, 3D vectors.
// Two separate defs are needed because the float and half variants alias
// different builtins (__builtin_hlsl_crossf32 vs __builtin_hlsl_crossf16).
def hlsl_cross_float : HLSLBuiltin<"cross", "__builtin_hlsl_crossf32"> {
  let Doc = [{
\fn T cross(T x, T y)
\brief Returns the cross product of two floating-point, 3D vectors.
\param x [in] The first floating-point, 3D vector.
\param y [in] The second floating-point, 3D vector.

Result is the cross product of x and y, i.e., the resulting
components are, in order :
x[1] * y[2] - y[1] * x[2]
x[2] * y[0] - y[2] * x[0]
x[0] * y[1] - y[0] * x[1]
}];
  let Args = [VectorType<FloatTy, 3>, VectorType<FloatTy, 3>];
  let ReturnType = VectorType<FloatTy, 3>;
}

def hlsl_cross_half : HLSLBuiltin<"cross", "__builtin_hlsl_crossf16"> {
  let Args = [VectorType<HalfTy, 3>, VectorType<HalfTy, 3>];
  let ReturnType = VectorType<HalfTy, 3>;
}

// Returns the dot product (a scalar value) of X and Y.
def hlsl_dot : HLSLTwoArgBuiltin<"dot", "__builtin_hlsl_dot"> {
  let Doc = [{
\fn K dot(T X, T Y)
\brief Return the dot product (a scalar value) of \a X and \a Y.
\param X The X input value.
\param Y The Y input value.
}];
  let ReturnType = VaryingElemType;
  let VaryingTypes = NumericTypesNoDbl;
  let VaryingMatDims = [];
}

// double dot only has scalar overload (no vectors).
def hlsl_dot_double : HLSLBuiltin<"dot", "__builtin_hlsl_dot"> {
  let Args = [Varying, Varying];
  let ReturnType = VaryingElemType;
  let VaryingTypes = [DoubleTy];
  let VaryingScalar = 1;
}

// Dot product of 2 half vectors plus a float scalar.
def hlsl_dot2add : HLSLBuiltin<"dot2add"> {
  let Doc = [{
\fn float dot2add(half2 A, half2 B, float C)
\brief Dot product of 2 vector of type half and add a float scalar value.
\param A The first input value to dot product.
\param B The second input value to dot product.
\param C The input value added to the dot product.
}];
  let DetailFunc = "dot2add_impl";
  let ParamNames = ["A", "B", "C"];
  let Args = [VectorType<HalfTy, 2>, VectorType<HalfTy, 2>, FloatTy];
  let ReturnType = FloatTy;
  let Availability = SM6_4;
}

// Blocks execution of all threads in a group until all group shared accesses
// have been completed and all threads in the group have reached this call.
def hlsl_group_memory_barrier_with_group_sync :
    HLSLBuiltin<"GroupMemoryBarrierWithGroupSync",
                "__builtin_hlsl_group_memory_barrier_with_group_sync"> {
  let Doc = [{
\fn void GroupMemoryBarrierWithGroupSync(void)
\brief Blocks execution of all threads in a group until all group shared
accesses have been completed and all threads in the group have reached this
call.
}];
  let IsConvergent = 1;
}

// Determines if the specified value x is infinite.
def hlsl_isinf : HLSLOneArgBuiltin<"isinf", "__builtin_hlsl_elementwise_isinf"> {
  let Doc = [{
\fn T isinf(T x)
\brief Determines if the specified value \a x is infinite.
\param x The specified input value.

Returns a value of the same size as the input, with a value set
to True if the x parameter is +INF or -INF. Otherwise, False.
}];
  let ReturnType = VaryingShape<BoolTy>;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns a refraction vector using an entering ray, a surface normal, and
// a refraction index.
def hlsl_refract : HLSLBuiltin<"refract"> {
  let Doc = [{
\fn T refract(T I, T N, T eta)
\brief Returns a refraction using an entering ray, \a I, a surface
normal, \a N and refraction index \a eta
\param I The entering ray.
\param N The surface normal.
\param eta The refraction index.

The return value is a floating-point vector that represents the refraction
using the refraction index, \a eta, for the direction of the entering ray,
\a I, off a surface with the normal \a N.

This function calculates the refraction vector using the following formulas:
k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I))
if k < 0.0 the result is 0.0
otherwise, the result is eta * I - (eta * dot(N, I) + sqrt(k)) * N

I and N must already be normalized in order to achieve the desired result.

I and N must be a scalar or vector whose component type is
floating-point.

eta must be a 16-bit or 32-bit floating-point scalar.

Result type, the type of I, and the type of N must all be the same type.
}];
  let DetailFunc = "refract_impl";
  let ParamNames = ["I", "N", "eta"];
  let Args = [Varying, Varying, VaryingElemType];
  let ReturnType = Varying;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = [];
}

