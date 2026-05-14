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

// The varying type - expanded per VaryingTypes.
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

  // Argument element types - drives overload expansion.
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

// Unsigned abs is a constexpr identity - unsigned values are already non-negative.
def hlsl_abs_unsigned : HLSLOneArgInlineBuiltin<"abs"> {
  let Doc = [{
\fn T abs(T Val)
\brief Returns the absolute value of the input value, \a Val.
\param Val The input value.

Unsigned overload - unsigned values are already non-negative, so this
function returns its input unchanged.
}];
  let ParamNames = ["V"];
  let Body = "return V;";
  let IsConstexpr = 1;
  let VaryingTypes = UnsignedIntTypes;
  let VaryingMatDims = [];
}

// Returns the arccosine of the input value, Val.
def hlsl_acos : HLSLOneArgBuiltin<"acos", "__builtin_elementwise_acos"> {
  let Doc = [{
\fn T acos(T Val)
\brief Returns the arccosine of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Implements unsigned 64-bit integer addition using pairs of unsigned 32-bit
// integers.
def hlsl_adduint64 : HLSLBuiltin<"AddUint64", "__builtin_hlsl_adduint64"> {
  let Doc = [{
\fn T AddUint64(T a, T b)
\brief Implements unsigned 64-bit integer addition using pairs of unsigned
32-bit integers.
\param x [in] The first unsigned 32-bit integer pair(s)
\param y [in] The second unsigned 32-bit integer pair(s)

This function takes one or two pairs (low, high) of unsigned 32-bit integer
values and returns pairs (low, high) of unsigned 32-bit integer
values representing the result of unsigned 64-bit integer addition.
}];
  let Args = [Varying, Varying];
  let ReturnType = Varying;
  let VaryingTypes = [UIntTy];
  let VaryingVecSizes = [2, 4];
  let Availability = SM6_0;
}

// Returns True if all components of the x parameter are non-zero;
// otherwise, false.
def hlsl_all : HLSLOneArgBuiltin<"all", "__builtin_hlsl_all"> {
  let Doc = [{
\fn bool all(T x)
\brief Returns True if all components of the \a x parameter are non-zero;
otherwise, false.
\param x The input value.
}];
  let ReturnType = BoolTy;
  let VaryingTypes = AllTypesWithBool;
  let VaryingMatDims = [];
}

// Returns the boolean AND of two boolean scalars, vectors, or matrices.
def hlsl_and : HLSLTwoArgBuiltin<"and", "__builtin_hlsl_and"> {
  let Doc = [{
\fn bool and(bool x, bool y)
\brief Logically ands two boolean vectors or matrices elementwise and
produces a bool vector or matrix output.
}];
  let VaryingTypes = [BoolTy];
}

// Returns True if any components of the x parameter are non-zero;
// otherwise, false.
def hlsl_any : HLSLOneArgBuiltin<"any", "__builtin_hlsl_any"> {
  let Doc = [{
\fn bool any(T x)
\brief Returns True if any components of the \a x parameter are non-zero;
otherwise, false.
\param x The input value.
}];
  let ReturnType = BoolTy;
  let VaryingTypes = AllTypesWithBool;
  let VaryingMatDims = [];
}

// Reinterprets a cast value (two 32-bit values) into a double.
def hlsl_asdouble : HLSLTwoArgBuiltin<"asdouble", "__builtin_hlsl_asdouble"> {
  let Doc = [{
\fn double asdouble(uint LowBits, uint HighBits)
\brief Reinterprets a cast value (two 32-bit values) into a double.
\param LowBits The low 32-bit pattern of the input value.
\param HighBits The high 32-bit pattern of the input value.
}];
  let ReturnType = VaryingShape<DoubleTy>;
  let VaryingTypes = [UIntTy];
  let VaryingMatDims = [];
}

// Returns the arcsine of the input value, Val.
def hlsl_asin : HLSLOneArgBuiltin<"asin", "__builtin_elementwise_asin"> {
  let Doc = [{
\fn T asin(T Val)
\brief Returns the arcsine of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the arctangent of the input value, Val.
def hlsl_atan : HLSLOneArgBuiltin<"atan", "__builtin_elementwise_atan"> {
  let Doc = [{
\fn T atan(T Val)
\brief Returns the arctangent of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the arctangent of y/x, using the signs of the arguments to determine
// the correct quadrant.
def hlsl_atan2 : HLSLTwoArgBuiltin<"atan2", "__builtin_elementwise_atan2"> {
  let Doc = [{
\fn T atan2(T y, T x)
\brief Returns the arctangent of y/x, using the signs of the arguments to
determine the correct quadrant.
\param y The y-coordinate.
\param x The x-coordinate.
}];
  let VaryingTypes = [HalfTy, FloatTy];
}

// Returns the smallest integer value that is greater than or equal to the
// input value, Val.
def hlsl_ceil : HLSLOneArgBuiltin<"ceil", "__builtin_elementwise_ceil"> {
  let Doc = [{
\fn T ceil(T Val)
\brief Returns the smallest integer value that is greater than or equal to
the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Checks whether the value is fully mapped.
def hlsl_check_access_fully_mapped : HLSLBuiltin<"CheckAccessFullyMapped"> {
  let Doc = [{
\fn bool CheckAccessFullyMapped(uint Status)
\brief Checks whether the value is fully mapped.
\param Status The status value to check.
}];
  let ParamNames = ["Status"];
  let Body = "return static_cast<bool>(Status);";
  let Args = [UIntTy];
  let ReturnType = BoolTy;
}

// Clamps the specified value X to the specified minimum (Min) and maximum (Max)
// range.
def hlsl_clamp : HLSLThreeArgBuiltin<"clamp", "__builtin_hlsl_elementwise_clamp"> {
  let Doc = [{
\fn T clamp(T X, T Min, T Max)
\brief Clamps the specified value \a X to the specified
minimum ( \a Min) and maximum ( \a Max) range.
\param X A value to clamp.
\param Min The specified minimum range.
\param Max The specified maximum range.

Returns The clamped value for the \a X parameter.
For values of -INF or INF, clamp will behave as expected.
However for values of NaN, the results are undefined.
}];
  let VaryingTypes = AllNumericTypes;
  let VaryingMatDims = [];
}

// Discards the current pixel if the specified value is less than zero.
def hlsl_clip : HLSLOneArgBuiltin<"clip", "__builtin_hlsl_elementwise_clip"> {
  let Doc = [{
\fn void clip(T Val)
\brief Discards the current pixel if the specified value is less than zero.
\param Val The input value.
}];
  let ReturnType = VoidTy;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the cosine of the input value, Val.
def hlsl_cos : HLSLOneArgBuiltin<"cos", "__builtin_elementwise_cos"> {
  let Doc = [{
\fn T cos(T Val)
\brief Returns the cosine of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the hyperbolic cosine of the input value, Val.
def hlsl_cosh : HLSLOneArgBuiltin<"cosh", "__builtin_elementwise_cosh"> {
  let Doc = [{
\fn T cosh(T Val)
\brief Returns the hyperbolic cosine of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the number of bits (per component) set in the input integer.
def hlsl_countbits : HLSLBuiltin<"countbits"> {
  let Doc = [{
\fn T countbits(T Val)
\brief Return the number of bits (per component) set in the input integer.
\param Val The input value.
}];
  let ParamNames = ["x"];
  let Body = "return __builtin_elementwise_popcount(x);";
  let Args = [Varying];
  let ReturnType = VaryingShape<UIntTy>;
  let VaryingTypes = AllIntTypes;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
}

// Returns the cross product of two floating-point, 3D vectors.
def hlsl_cross_float : HLSLBuiltin<"cross", "__builtin_hlsl_crossf32"> {
  let Args = [VectorType<FloatTy, 3>, VectorType<FloatTy, 3>];
  let ReturnType = VectorType<FloatTy, 3>;
}

def hlsl_cross_half : HLSLBuiltin<"cross", "__builtin_hlsl_crossf16"> {
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
  let Args = [VectorType<HalfTy, 3>, VectorType<HalfTy, 3>];
  let ReturnType = VectorType<HalfTy, 3>;
  let Availability = SM6_2;
}

// Converts a floating-point, 4D vector set by a D3DCOLOR to a UBYTE4.
def hlsl_d3d_color_to_ubyte4 : HLSLBuiltin<"D3DCOLORtoUBYTE4"> {
  let Doc = [{
\fn int4 D3DCOLORtoUBYTE4(float4 x)
\brief Converts a floating-point, 4D vector set by a D3DCOLOR to a UBYTE4.
\param x [in] The floating-point vector4 to convert.

The return value is the UBYTE4 representation of the \a x parameter.

This function swizzles and scales components of the \a x parameter. Use this
function to compensate for the lack of UBYTE4 support in some hardware.
}];
  // Use the scaling factor used by FXC, and DXC for DXIL (i.e., 255.001953).
  // The [DXC implementation] refers to [stack overflow] to justify the scaling
  // factor:
  // > Built-in rounding, necessary because of truncation. 0.001953 * 256 = 0.5
  // 
  // [DXC implementation]: https://github.com/microsoft/DirectXShaderCompiler/blob/070d0d5a2beacef9eeb51037a9b04665716fd6f3/lib/HLSL/HLOperationLower.cpp#L666C1-L697C2
  // [stack overflow]: https://stackoverflow.com/questions/52103720/why-does-d3dcolortoubyte4-multiplies-components-by-255-001953f
  let Body = "return V.zyxw * 255.001953f;";
  let ParamNames = ["V"];
  let Args = [VectorType<FloatTy, 4>];
  let ReturnType = VectorType<IntTy, 4>;
  let IsConstexpr = 1;
}

// Computes the partial derivative with regard to the x screen space coordinate.
def hlsl_ddx : HLSLOneArgDetail<"ddx", "ddx_impl"> {
  let Doc = [{
\fn T ddx(T x)
\brief Computes the partial derivative of the specified value with regard to
the screen-space x-coordinate.
\param x [in] The floating-point scalar or vector to process.
}];
  let ParamNames = ["input"];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Computes a low precision partial derivative with respect to the screen-space
// x-coordinate.
def hlsl_ddx_coarse : HLSLOneArgBuiltin<"ddx_coarse",
                                    "__builtin_hlsl_elementwise_ddx_coarse"> {
  let Doc = [{
\fn T ddx_coarse(T value)
\brief Computes a low precision partial derivative with respect to the
screen-space x-coordinate.
\param value The input value.

The return value is a floating point scalar or vector containing the low
prevision partial derivative of the input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Computes a high precision partial derivative with respect to the screen-space
// x-coordinate.
def hlsl_ddx_fine : HLSLOneArgBuiltin<"ddx_fine",
                                  "__builtin_hlsl_elementwise_ddx_fine"> {
  let Doc = [{
\fn T ddx_fine(T value)
\brief Computes a high precision partial derivative with respect to the
screen-space x-coordinate.
\param value The input value.

The return value is a floating point scalar or vector containing the high
prevision partial derivative of the input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Computes the partial derivative with regard to the y screen space coordinate.
def hlsl_ddy : HLSLOneArgDetail<"ddy", "ddy_impl"> {
  let Doc = [{
\fn T ddy(T x)
\brief Computes the partial derivative of the specified value with regard to
the screen-space y-coordinate.
\param x [in] The floating-point scalar or vector to process.
}];
  let ParamNames = ["input"];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Computes a low precision partial derivative with respect to the screen-space
// y-coordinate.
def hlsl_ddy_coarse : HLSLOneArgBuiltin<"ddy_coarse",
                                    "__builtin_hlsl_elementwise_ddy_coarse"> {
  let Doc = [{
\fn T ddy_coarse(T value)
\brief Computes a low precision partial derivative with respect to the
screen-space y-coordinate.
\param value The input value.

The return value is a floating point scalar or vector containing the low
prevision partial derivative of the input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Computes a high precision partial derivative with respect to the screen-space
// y-coordinate.
def hlsl_ddy_fine : HLSLOneArgBuiltin<"ddy_fine",
                                  "__builtin_hlsl_elementwise_ddy_fine"> {
  let Doc = [{
\fn T ddy_fine(T value)
\brief Computes a high precision partial derivative with respect to the
screen-space y-coordinate.
\param value The input value.

The return value is a floating point scalar or vector containing the high
prevision partial derivative of the input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Converts the specified value from radians to degrees.
def hlsl_degrees : HLSLOneArgBuiltin<"degrees", "__builtin_hlsl_elementwise_degrees"> {
  let Doc = [{
\fn T degrees(T x)
\brief Converts the specified value from radians to degrees.
\param x The specified input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns a distance scalar between X and Y.
// The distance between X and Y is length(X - Y).
def hlsl_distance : HLSLBuiltin<"distance"> {
  let Doc = [{
\fn K distance(T X, T Y)
\brief Returns a distance scalar between \a X and \a Y.
\param X The X input value.
\param Y The Y input value.
}];
  let Body = "return __detail::length_impl(X - Y);";
  let ParamNames = ["X", "Y"];
  let Args = [Varying, Varying];
  let ReturnType = VaryingElemType;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = [];
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
\fn float dot2add(half2 A, half2 B, float Acc)
\brief Multiplies the elements of the two half-precision float input vectors
together and sums the results into the 32-bit float accumulator.
\param A The first input value to dot product.
\param B The second input value to dot product.
\param Acc The accumulator value added to the dot product.

This instruction operates within a single 32-bit wide SIMD lane.
The inputs are 16-bit quantities packed into the same lane.
}];
  let DetailFunc = "dot2add_impl";
  let ParamNames = ["A", "B", "Acc"];
  let Args = [VectorType<HalfTy, 2>, VectorType<HalfTy, 2>, FloatTy];
  let ReturnType = FloatTy;
  let Availability = SM6_4;
}

// Dot product of two packed 8-bit signed integer vectors, accumulated with a
// 32-bit signed integer.
def hlsl_dot4add_i8packed :
    HLSLBuiltin<"dot4add_i8packed", "__builtin_hlsl_dot4add_i8packed"> {
  let Doc = [{
\fn int dot4add_i8packed(uint A, uint B, int C)
\brief Dot product of two packed 8-bit signed integer vectors, accumulated
with a 32-bit signed integer.
}];
  let Args = [UIntTy, UIntTy, IntTy];
  let ReturnType = IntTy;
  let Availability = SM6_4;
}

// Dot product of two packed 8-bit unsigned integer vectors, accumulated with a
// 32-bit unsigned integer.
def hlsl_dot4add_u8packed :
    HLSLBuiltin<"dot4add_u8packed", "__builtin_hlsl_dot4add_u8packed"> {
  let Doc = [{
\fn uint dot4add_u8packed(uint A, uint B, uint C)
\brief Dot product of two packed 8-bit unsigned integer vectors, accumulated
with a 32-bit unsigned integer.
}];
  let Args = [UIntTy, UIntTy, UIntTy];
  let ReturnType = UIntTy;
  let Availability = SM6_4;
}

// Calculates a distance vector.
def hlsl_dst : HLSLBuiltin<"dst"> {
  let Doc = [{
\fn vector<T, 4> dst(vector<T, 4> Src0, vector<T, 4> Src1)
\brief Calculates a distance vector.
\param Src0 [in] Contains the squared distance.
\param Src1 [in] Contains the reciprocal distance.

Return the computed distance vector.
}];
  let ParamNames = ["Src0", "Src1"];
  let Body = "return {1, Src0[1] * Src1[1], Src0[2], Src1[3]};";
  let Args = [Varying, Varying];
  let ReturnType = Varying;
  let VaryingTypes = AllFloatTypes;
  let VaryingVecSizes = [4];
  let VaryingMatDims = [];
}

// Returns the base-e exponential, or e**x, of the specified value.
def hlsl_exp : HLSLOneArgBuiltin<"exp", "__builtin_elementwise_exp"> {
  let Doc = [{
\fn T exp(T x)
\brief Returns the base-e exponential, or \a e**x, of the specified value.
\param x The specified input value.

The return value is the base-e exponential of the \a x parameter.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the base 2 exponential, or 2**x, of the specified value.
def hlsl_exp2 : HLSLOneArgBuiltin<"exp2", "__builtin_elementwise_exp2"> {
  let Doc = [{
\fn T exp2(T x)
\brief Returns the base 2 exponential, or \a 2**x, of the specified value.
\param x The specified input value.

The base 2 exponential of the \a x parameter.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the half value stored in the low 16 bits of the uint arg converted
// to a float.
def hlsl_f16tof32 : HLSLOneArgBuiltin<"f16tof32",
    "__builtin_hlsl_elementwise_f16tof32"> {
  let Doc = [{
\fn float f16tof32(uint x)
\brief Returns the half value stored in the low 16 bits of the uint arg
converted to a float.
\param x The uint containing two half values.

The float value of the half value found in the low 16 bits of the \a x
parameter.
}];
  let ReturnType = VaryingShape<FloatTy>;
  let VaryingTypes = [UIntTy];
  let VaryingMatDims = [];
}

// Returns the float arg value converted to half in the low 16 bits of the uint
// return value.
def hlsl_f32tof16 : HLSLOneArgBuiltin<"f32tof16",
    "__builtin_hlsl_elementwise_f32tof16"> {
  let Doc = [{
\fn uint f32tof16(float x)
\brief Returns the float arg value converted to half in the low 16 bits of
the uint return value.
\param x The float to be converted to half.

The return value is a uint containing the converted half value in the low
16 bits.
}];
  let ReturnType = VaryingShape<UIntTy>;
  let VaryingTypes = [FloatTy];
  let VaryingMatDims = [];
}

// Flips the surface-normal to face in a direction opposite to I.
def hlsl_faceforward : HLSLThreeArgDetail<"faceforward", "faceforward_impl"> {
  let Doc = [{
\fn T faceforward(T N, T I, T Ng)
\brief Flips the surface-normal (if needed) to face in a direction opposite
to \a I. Returns the result in terms of \a N.
\param N The resulting floating-point surface-normal vector.
\param I A floating-point, incident vector that points from the view
position to the shading position.
\param Ng A floating-point surface-normal vector.

Return a floating-point, surface normal vector that is facing the view
direction.
}];
  let ParamNames = ["N", "I", "Ng"];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Return fused multiply-add of double-precision arguments: a * b + c.
def hlsl_fma : HLSLThreeArgBuiltin<"fma", "__builtin_elementwise_fma"> {
  let Doc = [{
\fn double fma(double a, double b, double c)
\brief Returns the double-precision fused multiply-addition of a * b + c.
\param a The first value in the fused multiply-addition.
\param b The second value in the fused multiply-addition.
\param c The third value in the fused multiply-addition.
}];
  let VaryingTypes = [DoubleTy];
}

// Returns the location of the first set bit starting from the lowest order bit
// and working upward, per component.
def hlsl_firstbitlow : HLSLOneArgBuiltin<"firstbitlow",
                                    "__builtin_hlsl_elementwise_firstbitlow"> {
  let Doc = [{
\fn T firstbitlow(T Val)
\brief Returns the location of the first set bit starting from the lowest
order bit and working upward, per component.
\param Val the input value.
}];
  let ReturnType = VaryingShape<UIntTy>;
  let VaryingTypes = AllIntTypes;
  let VaryingMatDims = [];
}

// Returns the largest integer that is less than or equal to the input
// value, Val.
def hlsl_floor : HLSLOneArgBuiltin<"floor", "__builtin_elementwise_floor"> {
  let Doc = [{
\fn T floor(T Val)
\brief Returns the largest integer that is less than or equal to the input
value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the floating-point remainder of x/y (scalar overloads).
def hlsl_fmod : HLSLBuiltin<"fmod"> {
  let Doc = [{
\fn T fmod(T x, T y)
\brief Returns the floating-point remainder of x/y.
\param x [in] The dividend.
\param y [in] The divisor.

Return the floating-point remainder of the x parameter divided by the y
parameter.
}];
  let DetailFunc = "fmod_impl";
  let ParamNames = ["X", "Y"];
  let Args = [Varying, Varying];
  let ReturnType = Varying;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingScalar = 1;
  let VaryingMatDims = [];
}

// Returns the floating-point remainder of x/y (vector overloads).
def hlsl_fmod_vec : HLSLBuiltin<"fmod"> {
  let DetailFunc = "fmod_vec_impl";
  let ParamNames = ["X", "Y"];
  let Args = [Varying, Varying];
  let ReturnType = Varying;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = [];
}

// Returns the fractional (or decimal) part of x.
def hlsl_frac : HLSLOneArgBuiltin<"frac", "__builtin_hlsl_elementwise_frac"> {
  let Doc = [{
\fn T frac(T x)
\brief Returns the fractional (or decimal) part of x. \a x parameter.
\param x The specified input value.

If \a the return value is greater than or equal to 0 and less than 1.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Computes the sum of the absolute values of the partial derivatives.
def hlsl_fwidth : HLSLOneArgDetail<"fwidth", "fwidth_impl"> {
  let Doc = [{
\fn T fwidth(T x)
\brief Computes the sum of the absolute values of the partial derivatives
with regard to the x and y screen space coordinates.
\param x [in] The floating-point scalar or vector to process.
}];
  let ParamNames = ["input"];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Blocks execution of all threads in a group until all memory accesses
// have been completed.
def hlsl_all_memory_barrier :
    HLSLBuiltin<"AllMemoryBarrier",
                "__builtin_hlsl_all_memory_barrier"> {
  let Doc = [{
\fn void AllMemoryBarrier(void)
\brief Blocks execution of all threads in a group until all memory
accesses have been completed.
}];
  let IsConvergent = 1;
}

// Blocks execution of all threads in a group until all memory accesses
// have been completed and all threads in the group have reached this call.
def hlsl_all_memory_barrier_with_group_sync :
    HLSLBuiltin<"AllMemoryBarrierWithGroupSync",
                "__builtin_hlsl_all_memory_barrier_with_group_sync"> {
  let Doc = [{
\fn void AllMemoryBarrierWithGroupSync(void)
\brief Blocks execution of all threads in a group until all memory
accesses have been completed and all threads in the group have reached this call.
}];
  let IsConvergent = 1;
}

// Blocks execution of all threads in a group until all device memory accesses
// have been completed.
def hlsl_device_memory_barrier :
    HLSLBuiltin<"DeviceMemoryBarrier",
                "__builtin_hlsl_device_memory_barrier"> {
  let Doc = [{
\fn void DeviceMemoryBarrier(void)
\brief Blocks execution of all threads in a group until all device memory
accesses have been completed.
}];
  let IsConvergent = 1;
}

// Blocks execution of all threads in a group until all device memory accesses
// have been completed and all threads in the group have reached this call.
def hlsl_device_memory_barrier_with_group_sync :
    HLSLBuiltin<"DeviceMemoryBarrierWithGroupSync",
                "__builtin_hlsl_device_memory_barrier_with_group_sync"> {
  let Doc = [{
\fn void DeviceMemoryBarrierWithGroupSync(void)
\brief Blocks execution of all threads in a group until all device memory
accesses have been completed and all threads in the group have reached this call.
}];
  let IsConvergent = 1;
}

// Blocks execution of all threads in a group until all group shared accesses
// have been completed.
def hlsl_group_memory_barrier :
    HLSLBuiltin<"GroupMemoryBarrier",
                "__builtin_hlsl_group_memory_barrier"> {
  let Doc = [{
\fn void GroupMemoryBarrier(void)
\brief Blocks execution of all threads in a group until all group shared
accesses have been completed.
}];
  let IsConvergent = 1;
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
\brief Determines if the specified value \a x  is infinite.
\param x The specified input value.

Returns a value of the same size as the input, with a value set
to True if the x parameter is +INF or -INF. Otherwise, False.
}];
  let ReturnType = VaryingShape<BoolTy>;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Determines if the specified value x is Not a Number.
def hlsl_isnan : HLSLOneArgBuiltin<"isnan", "__builtin_hlsl_elementwise_isnan"> {
  let Doc = [{
\fn T isnan(T x)
\brief Determines if the specified value \a x  is Not a Number.
\param x The specified input value.

Returns a value of the same size as the input, with a value set
to True if the x parameter is NaN or QNaN. Otherwise, False.
}];
  let ReturnType = VaryingShape<BoolTy>;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the result of multiplying the specified value by two raised
// to the power of the specified exponent.
def hlsl_ldexp : HLSLBuiltin<"ldexp"> {
  let Doc = [{
\fn T ldexp(T X, T Exp)
\brief Returns the result of multiplying the specified value by two raised
to the power of the specified exponent.
\param X [in] The specified value.
\param Exp [in] The specified exponent.

This function uses the following formula: X * 2^Exp
}];
  // ldexp(X, Exp) = X * 2^Exp, implemented as exp2(Exp) * X.
  let ParamNames = ["X", "Exp"];
  let Body = "return exp2(Exp) * X;";
  let Args = [Varying, Varying];
  let ReturnType = Varying;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = [];
}

// Returns the length of the specified floating-point vector.
// Scalar length is equivalent to abs(X).
// Vector length is equivalent to sqrt(dot(X, X)).
def hlsl_length : HLSLBuiltin<"length"> {
  let Doc = [{
\fn T length(T x)
\brief Returns the length of the specified floating-point vector.
\param x [in] The vector of floats, or a scalar float.

Length is based on the following formula: sqrt(x[0]^2 + x[1]^2 + ...).
}];
  let DetailFunc = "length_impl";
  let ParamNames = ["X"];
  let Args = [Varying];
  let ReturnType = VaryingElemType;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = [];
}

// Returns the linear interpolation of x to y by s.
def hlsl_lerp : HLSLThreeArgBuiltin<"lerp", "__builtin_hlsl_lerp"> {
  let Doc = [{
\fn T lerp(T x, T y, T s)
\brief Returns the linear interpolation of x to y by s.
\param x [in] The first-floating point value.
\param y [in] The second-floating point value.
\param s [in] A value that linearly interpolates between the x parameter and
the y parameter.

Linear interpolation is based on the following formula: x*(1-s) + y*s which
can equivalently be written as x + s(y-x).
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

def hlsl_lit_float : HLSLBuiltin<"lit"> {
  let DetailFunc = "lit_impl";
  let ParamNames = ["NDotL", "NDotH", "M"];
  let Args = [FloatTy, FloatTy, FloatTy];
  let ReturnType = VectorType<FloatTy, 4>;
}

// Returns a lighting coefficient vector.
def hlsl_lit_half : HLSLBuiltin<"lit"> {
  let Doc = [{
\fn vector<T, 4> lit(T NDotL, T NDotH, T M)
\brief Returns a lighting coefficient vector.
\param NDotL The dot product of the normalized surface normal and the
light vector.
\param NDotH The dot product of the half-angle vector and the surface
normal.
\param M A specular exponent.

This function returns a lighting coefficient vector (ambient, diffuse,
specular, 1).
}];
  let DetailFunc = "lit_impl";
  let ParamNames = ["NDotL", "NDotH", "M"];
  let Args = [HalfTy, HalfTy, HalfTy];
  let ReturnType = VectorType<HalfTy, 4>;
}

// Returns the base-e logarithm of the input value, Val.
def hlsl_log : HLSLOneArgBuiltin<"log", "__builtin_elementwise_log"> {
  let Doc = [{
\fn T log(T Val)
\brief The base-e logarithm of the input value, \a Val parameter.
\param Val The input value.

If \a Val is negative, this result is undefined. If \a Val is 0, this
function returns negative infinity.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the base-10 logarithm of the input value, Val.
def hlsl_log10 : HLSLOneArgBuiltin<"log10", "__builtin_elementwise_log10"> {
  let Doc = [{
\fn T log10(T Val)
\brief The base-10 logarithm of the input value, \a Val parameter.
\param Val The input value.

If \a Val is negative, this result is undefined. If \a Val is 0, this
function returns negative infinity.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the base-2 logarithm of the input value, Val.
def hlsl_log2 : HLSLOneArgBuiltin<"log2", "__builtin_elementwise_log2"> {
  let Doc = [{
\fn T log2(T Val)
\brief The base-2 logarithm of the input value, \a Val parameter.
\param Val The input value.

If \a Val is negative, this result is undefined. If \a Val is 0, this
function returns negative infinity.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the result of M * A + B (multiply-add).
def hlsl_mad : HLSLThreeArgBuiltin<"mad", "__builtin_hlsl_mad"> {
  let Doc = [{
\fn T mad(T M, T A, T B)
\brief The result of \a M * \a A + \a B.
\param M The multiplication value.
\param A The first addition value.
\param B The second addition value.
}];
  let VaryingTypes = AllNumericTypes;
  let VaryingMatDims = [];
}

// Returns the greater of X and Y.
def hlsl_max : HLSLTwoArgBuiltin<"max", "__builtin_elementwise_max"> {
  let Doc = [{
\fn T max(T X, T Y)
\brief Return the greater of \a X and \a Y.
\param X The X input value.
\param Y The Y input value.
}];
  let VaryingTypes = AllNumericTypes;
  let VaryingMatDims = [];
}

// Returns the lesser of X and Y.
def hlsl_min : HLSLTwoArgBuiltin<"min", "__builtin_elementwise_min"> {
  let Doc = [{
\fn T min(T X, T Y)
\brief Return the lesser of \a X and \a Y.
\param X The X input value.
\param Y The Y input value.
}];
  let VaryingTypes = AllNumericTypes;
  let VaryingMatDims = [];
}

// Marks a resource index as non-uniform (varying across wave threads).
def hlsl_non_uniform_resource_index :
    HLSLBuiltin<"NonUniformResourceIndex",
                "__builtin_hlsl_resource_nonuniformindex"> {
  let Doc = [{
\fn uint NonUniformResourceIndex(uint I)
\brief A compiler hint to indicate that a resource index varies across
threads within a wave (i.e., it is non-uniform).
\param I [in] Resource array index

The return value is the \a I parameter.
}];
  let Args = [UIntTy];
  let ReturnType = UIntTy;
}

// Returns the normalized unit vector of the specified floating-point vector.
def hlsl_normalize : HLSLOneArgBuiltin<"normalize", "__builtin_hlsl_normalize"> {
  let Doc = [{
\fn T normalize(T x)
\brief Returns the normalized unit vector of the specified floating-point
vector.
\param x [in] The vector of floats.

Normalize is based on the following formula: x / length(x).
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the boolean OR of two boolean scalars, vectors, or matrices.
def hlsl_or : HLSLTwoArgBuiltin<"or", "__builtin_hlsl_or"> {
  let Doc = [{
\fn bool or(bool x, bool y)
\brief Logically ors two boolean vectors or matrices elementwise and
 produces a bool vector or matrix output.
}];
  let VaryingTypes = [BoolTy];
}

// Returns the value Val, raised to the power Pow.
def hlsl_pow : HLSLTwoArgBuiltin<"pow", "__builtin_elementwise_pow"> {
  let Doc = [{
\fn T pow(T Val, T Pow)
\brief Return the value \a Val, raised to the power \a Pow.
\param Val The input value.
\param Pow The specified power.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Reads the value from the lane across the X axis of the quad.
def hlsl_quad_read_across_x :
    HLSLOneArgBuiltin<"QuadReadAcrossX",
                      "__builtin_hlsl_quad_read_across_x"> {
  let Doc = [{
\brief Reads the value from the lane across the X axis of the quad.
\param Val The input value.
}];
  let VaryingTypes = AllNumericTypes;
  let IsConvergent = 1;
  let VaryingMatDims = [];
}

// Reads the value from the lane across the X axis of the quad.
def hlsl_quad_read_across_y :
    HLSLOneArgBuiltin<"QuadReadAcrossY",
                      "__builtin_hlsl_quad_read_across_y"> {
  let Doc = [{
\brief Reads the value from the lane across the Y axis of the quad.
\param Val The input value.
}];
  let VaryingTypes = AllNumericTypes;
  let IsConvergent = 1;
  let VaryingMatDims = [];
}

// Converts the specified value from degrees to radians.
def hlsl_radians : HLSLOneArgBuiltin<"radians", "__builtin_hlsl_elementwise_radians"> {
  let Doc = [{
\fn T radians(T Val)
\brief Converts the specified value from degrees to radians.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Calculates a fast, approximate, per-component reciprocal ie 1 / x.
def hlsl_rcp : HLSLOneArgBuiltin<"rcp", "__builtin_hlsl_elementwise_rcp"> {
  let Doc = [{
\fn T rcp(T x)
\brief Calculates a fast, approximate, per-component reciprocal ie 1 / \a x.
\param x The specified input value.

The return value is the reciprocal of the \a x parameter.
}];
  let VaryingTypes = AllFloatTypes;
  let VaryingMatDims = [];
}

// Returns a reflection vector.
// Scalar reflect: V = I - 2 * N * I * N (dot product is just multiplication
// for scalars).
def hlsl_reflect : HLSLTwoArgDetail<"reflect", "reflect_impl"> {
  let Doc = [{
\fn T reflect(T I, T N)
\brief Returns a reflection using an incident ray, \a I, and a surface
normal, \a N.
\param I The incident ray.
\param N The surface normal.

The return value is a floating-point vector that represents the reflection
of the incident ray, \a I, off a surface with the normal \a N.

This function calculates the reflection vector using the following formula:
V = I - 2 * N * dot(I N) .

N must already be normalized in order to achieve the desired result.
}];
  let ParamNames = ["I", "N"];
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

// Returns the value Val with the bit order reversed.
def hlsl_reversebits : HLSLOneArgBuiltin<"reversebits", "__builtin_elementwise_bitreverse"> {
  let Doc = [{
\fn T reversebits(T Val)
\brief Return the value \a Val with the bit order reversed.
\param Val The input value.
}];
  let VaryingTypes = AllIntTypes;
  let VaryingMatDims = [];
}

// Rounds the specified value x to the nearest integer.
def hlsl_round : HLSLOneArgBuiltin<"round", "__builtin_elementwise_roundeven"> {
  let Doc = [{
\fn T round(T x)
\brief Rounds the specified value \a x to the nearest integer.
\param x The specified input value.

The return value is the \a x parameter, rounded to the nearest integer
within a floating-point type. Halfway cases are
rounded to the nearest even value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the reciprocal of the square root of the specified value,
// ie 1 / sqrt(x).
def hlsl_rsqrt : HLSLOneArgBuiltin<"rsqrt", "__builtin_hlsl_elementwise_rsqrt"> {
  let Doc = [{
\fn T rsqrt(T x)
\brief Returns the reciprocal of the square root of the specified value.
ie 1 / sqrt( \a x).
\param x The specified input value.

This function uses the following formula: 1 / sqrt(x).
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns input value, Val, clamped within the range of 0.0f to 1.0f.
def hlsl_saturate : HLSLOneArgBuiltin<"saturate", "__builtin_hlsl_elementwise_saturate"> {
  let Doc = [{
\fn T saturate(T Val)
\brief Returns input value, \a Val, clamped within the range of 0.0f
to 1.0f.
\param Val The input value.
}];
  let VaryingTypes = AllFloatTypes;
  let VaryingMatDims = [];
}

// Returns -1 if Val is less than zero; 0 if Val equals zero; and 1 if Val is
// greater than zero.
def hlsl_sign : HLSLOneArgBuiltin<"sign", "__builtin_hlsl_elementwise_sign"> {
  let Doc = [{
\fn T sign(T Val)
\brief Returns -1 if \a Val is less than zero; 0 if \a Val equals zero; and
1 if \a Val is greater than zero.
\param Val The input value.
}];
  let ReturnType = VaryingShape<IntTy>;
  let VaryingTypes = AllNumericTypes;
  let VaryingMatDims = [];
}

// Returns the sine of the input value, Val.
def hlsl_sin : HLSLOneArgBuiltin<"sin", "__builtin_elementwise_sin"> {
  let Doc = [{
\fn T sin(T Val)
\brief Returns the sine of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the hyperbolic sine of the input value, Val.
def hlsl_sinh : HLSLOneArgBuiltin<"sinh", "__builtin_elementwise_sinh"> {
  let Doc = [{
\fn T sinh(T Val)
\brief Returns the hyperbolic sine of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns a smooth Hermite interpolation between 0 and 1.
def hlsl_smoothstep : HLSLThreeArgDetail<"smoothstep", "smoothstep_impl"> {
  let Doc = [{
\fn T smoothstep(T Min, T Max, T X)
\brief Returns a smooth Hermite interpolation between 0 and 1, if \a X is in
the range [\a Min, \a Max].
\param Min The minimum range of the x parameter.
\param Max The maximum range of the x parameter.
\param X The specified value to be interpolated.

The return value is 0.0 if \a X <= \a Min and 1.0 if \a X >= \a Max. When \a
Min < \a X < \a Max, the function performs smooth Hermite interpolation
between 0 and 1.
}];
  let ParamNames = ["Min", "Max", "X"];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the square root of the input value, Val.
def hlsl_sqrt : HLSLOneArgBuiltin<"sqrt", "__builtin_elementwise_sqrt"> {
  let Doc = [{
\fn T sqrt(T Val)
\brief Returns the square root of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns 1 if the x parameter is greater than or equal to the y parameter;
// otherwise, 0.
def hlsl_step : HLSLTwoArgBuiltin<"step", "__builtin_hlsl_step"> {
  let Doc = [{
\fn T step(T y, T x)
\brief Returns 1 if the x parameter is greater than or equal to the y
parameter; otherwise, 0.
\param y [in] The first floating-point value to compare.
\param x [in] The second floating-point value to compare.

Step is based on the following formula: (x >= y) ? 1 : 0
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the tangent of the input value, Val.
def hlsl_tan : HLSLOneArgBuiltin<"tan", "__builtin_elementwise_tan"> {
  let Doc = [{
\fn T tan(T Val)
\brief Returns the tangent of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the hyperbolic tangent of the input value, Val.
def hlsl_tanh : HLSLOneArgBuiltin<"tanh", "__builtin_elementwise_tanh"> {
  let Doc = [{
\fn T tanh(T Val)
\brief Returns the hyperbolic tangent of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Returns the truncated integer value of the input value, Val.
def hlsl_trunc : HLSLOneArgBuiltin<"trunc", "__builtin_elementwise_trunc"> {
  let Doc = [{
\fn T trunc(T Val)
\brief Returns the truncated integer value of the input value, \a Val.
\param Val The input value.
}];
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingMatDims = [];
}

// Evaluates a value for all active invocations in the group. The result is true
// if Value is equal for all active invocations in the group.
def hlsl_wave_active_all_equal :
    HLSLOneArgBuiltin<"WaveActiveAllEqual",
                "__builtin_hlsl_wave_active_all_equal"> {
  let Doc = [{
\brief Evaluates a value for all active invocations in the group. The
result is true if Value is equal for all active invocations in the
group. Otherwise, the result is false.
\param Value The value to compare with
\return True if all values across all lanes are equal, false otherwise
}];
  let ReturnType = VaryingShape<BoolTy>;
  let VaryingTypes = AllNumericTypes;
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Returns true if the expression is true in all active lanes in the current
// wave.
def hlsl_wave_active_all_true :
    HLSLBuiltin<"WaveActiveAllTrue", "__builtin_hlsl_wave_active_all_true"> {
  let Doc = [{
\brief Returns true if the expression is true in all active lanes in the
current wave.
\param Val The boolean expression to evaluate.
\return True if the expression is true in all lanes.
}];
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingTypes = [BoolTy];
  let VaryingScalar = 1;
  let IsConvergent = 1;
  let Availability = SM6_0;
}

// Returns true if the expression is true in any active lane in the current
// wave.
def hlsl_wave_active_any_true :
    HLSLBuiltin<"WaveActiveAnyTrue", "__builtin_hlsl_wave_active_any_true"> {
  let Doc = [{
\brief Returns true if the expression is true in any active lane in the
current wave.
\param Val The boolean expression to evaluate.
\return True if the expression is true in any lane.
}];
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingTypes = [BoolTy];
  let VaryingScalar = 1;
  let IsConvergent = 1;
  let Availability = SM6_0;
}

// Returns a uint4 containing a bitmask of the evaluation of the boolean
// expression for all active lanes in the current wave.
def hlsl_wave_active_ballot :
    HLSLBuiltin<"WaveActiveBallot", "__builtin_hlsl_wave_active_ballot"> {
  let Doc = [{
\brief Returns a uint4 containing a bitmask of the evaluation of the
boolean expression for all active lanes in the current wave.
The least-significant bit corresponds to the lane with index zero.
The bits corresponding to inactive lanes will be zero. The bits that
are greater than or equal to WaveGetLaneCount will be zero.
\param Val The boolean expression to evaluate.
\return uint4 bitmask
}];
  let Args = [Varying];
  let ReturnType = VectorType<UIntTy, 4>;
  let VaryingTypes = [BoolTy];
  let VaryingScalar = 1;
  let IsConvergent = 1;
  let Availability = SM6_0;
}

// Returns the bitwise AND of all values across all active non-helper lanes.
def hlsl_wave_active_bit_and :
    HLSLOneArgBuiltin<"WaveActiveBitAnd",
                      "__builtin_hlsl_wave_active_bit_and"> {
  let VaryingTypes = [UIntTy, UInt64Ty];
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Returns the bitwise OR of all values across all active non-helper lanes.
def hlsl_wave_active_bit_or :
    HLSLOneArgBuiltin<"WaveActiveBitOr",
                      "__builtin_hlsl_wave_active_bit_or"> {
  let Doc = [{
\brief Returns the bitwise OR of all the values of <expr> across all active
non-helper lanes in the current wave, and replicates it back to
all active non-helper lanes.
}];
  let VaryingTypes = [UIntTy, UInt64Ty];
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Returns the bitwise XOR of all values across all active non-helper lanes.
def hlsl_wave_active_bit_xor :
    HLSLOneArgBuiltin<"WaveActiveBitXor",
                      "__builtin_hlsl_wave_active_bit_xor"> {
  let VaryingTypes = [UIntTy, UInt64Ty];
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Counts the number of boolean variables which evaluate to true across all
// active lanes in the current wave.
def hlsl_wave_active_count_bits :
    HLSLBuiltin<"WaveActiveCountBits",
                "__builtin_hlsl_wave_active_count_bits"> {
  let Doc = [{
\brief Counts the number of boolean variables which evaluate to true across
all active lanes in the current wave.
\param Val The input boolean value.
\return The number of lanes for which the boolean variable evaluates to
true, across all active lanes in the current wave.
}];
  let Args = [Varying];
  let ReturnType = VaryingShape<UIntTy>;
  let VaryingTypes = [BoolTy];
  let VaryingScalar = 1;
  let IsConvergent = 1;
  let Availability = SM6_0;
}

// Returns the maximum value across all active lanes in the wave.
def hlsl_wave_active_max :
    HLSLOneArgBuiltin<"WaveActiveMax", "__builtin_hlsl_wave_active_max"> {
  let Doc = [{
\brief Returns the maximum value across all active lanes in the wave.
\param Val The input value.
}];
  let VaryingTypes = AllNumericTypes;
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Returns the minimum value across all active lanes in the wave.
def hlsl_wave_active_min :
    HLSLOneArgBuiltin<"WaveActiveMin", "__builtin_hlsl_wave_active_min"> {
  let Doc = [{
\brief Returns the minimum value across all active lanes in the wave.
\param Val The input value.
}];
  let VaryingTypes = AllNumericTypes;
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Returns the product of values across all active lanes in the wave.
def hlsl_wave_active_product :
    HLSLOneArgBuiltin<"WaveActiveProduct",
                      "__builtin_hlsl_wave_active_product"> {
  let Doc = [{
\brief Returns the product of values across all active lanes in the wave.
\param Val The input value.
}];
  let VaryingTypes = AllNumericTypes;
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Returns the sum of values across all active lanes in the wave.
def hlsl_wave_active_sum :
    HLSLOneArgBuiltin<"WaveActiveSum", "__builtin_hlsl_wave_active_sum"> {
  let Doc = [{
\brief Returns the sum of values across all active lanes in the wave.
\param Val The input value.
}];
  let VaryingTypes = AllNumericTypes;
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Returns the number of lanes in the current wave.
def hlsl_wave_get_lane_count :
    HLSLBuiltin<"WaveGetLaneCount", "__builtin_hlsl_wave_get_lane_count"> {
  let Doc = [{
\brief Returns the number of lanes in the current wave.
}];
  let ReturnType = UIntTy;
  let IsConvergent = 1;
  let Availability = SM6_0;
}

// Returns the index of the current lane within the current wave.
def hlsl_wave_get_lane_index :
    HLSLBuiltin<"WaveGetLaneIndex", "__builtin_hlsl_wave_get_lane_index"> {
  let Doc = [{
\brief Returns the index of the current lane within the current wave.
}];
  let ReturnType = UIntTy;
  let IsConvergent = 1;
  let Availability = SM6_0;
}

// Returns true if the current lane is the first active lane in the wave.
def hlsl_wave_is_first_lane :
    HLSLBuiltin<"WaveIsFirstLane", "__builtin_hlsl_wave_is_first_lane"> {
  let Doc = [{
\brief Returns true if the current lane is the first active lane in the wave.
}];
  let ReturnType = BoolTy;
  let IsConvergent = 1;
  let Availability = SM6_0;
}

// Returns the count of bits set to 1 on prior lanes.
def hlsl_wave_prefix_count_bits :
    HLSLBuiltin<"WavePrefixCountBits",
                "__builtin_hlsl_wave_prefix_count_bits"> {
  let Doc = [{
\brief Returns the count of bits of Expr set to 1 on prior lanes.
\param Expr The boolean expression to evaluate.
\return the count of bits set to 1 on prior lanes.
}];
  let Args = [Varying];
  let ReturnType = VaryingShape<UIntTy>;
  let VaryingTypes = [BoolTy];
  let VaryingScalar = 1;
  let IsConvergent = 1;
  let Availability = SM6_0;
}

// Returns the product of values across lanes with smaller indices in the wave.
def hlsl_wave_prefix_product :
    HLSLOneArgBuiltin<"WavePrefixProduct", "__builtin_hlsl_wave_prefix_product"> {
  let Doc = [{
\brief Returns the product of values across lanes with smaller indices in the wave.
\param Val The input value.
}];
  let VaryingTypes = AllNumericTypes;
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Returns the sum of values across lanes with smaller indices in the wave.
def hlsl_wave_prefix_sum :
    HLSLOneArgBuiltin<"WavePrefixSum", "__builtin_hlsl_wave_prefix_sum"> {
  let Doc = [{
\brief Returns the sum of values across lanes with smaller indices in the wave.
\param Val The input value.
}];
  let VaryingTypes = AllNumericTypes;
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}

// Reads the value from the specified lane index in the wave.
def hlsl_wave_read_lane_at :
    HLSLTwoArgBuiltin<"WaveReadLaneAt", "__builtin_hlsl_wave_read_lane_at"> {
  let Doc = [{
\brief Returns the value of the expression for the given lane index within
the specified wave.
\param Val The value to read.
\param Index The lane index.
}];
  let Args = [Varying, UIntTy];
  let VaryingTypes = AllTypesWithBool;
  let IsConvergent = 1;
  let Availability = SM6_0;
  let VaryingMatDims = [];
}
