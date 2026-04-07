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

def hlsl_abs_unsigned : HLSLOneArgInlineBuiltin<"abs"> {
  let Doc = [{
\fn T abs(T Val)
\brief Returns the absolute value of the input value, \a Val.
\param Val The input value.

Unsigned overload — unsigned values are already non-negative, so this
function returns its input unchanged.
}];
  let ParamNames = ["V"];
  let Body = "return V;";
  let IsConstexpr = 1;
  let VaryingTypes = UnsignedIntTypes;
  let VaryingMatDims = [];
}

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
def hlsl_fmod_vec : HLSLBuiltin<"fmod"> {
  let DetailFunc = "fmod_vec_impl";
  let ParamNames = ["X", "Y"];
  let Args = [Varying, Varying];
  let ReturnType = Varying;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = [];
}

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

def hlsl_ldexp : HLSLBuiltin<"ldexp"> {
  let Doc = [{
\fn T ldexp(T X, T Exp)
\brief Returns the result of multiplying the specified value by two raised
to the power of the specified exponent.
\param X [in] The specified value.
\param Exp [in] The specified exponent.
}];
  let ParamNames = ["X", "Exp"];
  let Body = "return exp2(Exp) * X;";
  let Args = [Varying, Varying];
  let ReturnType = Varying;
  let VaryingTypes = [HalfTy, FloatTy];
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3, 4];
  let VaryingMatDims = [];
}

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

def hlsl_lit_float : HLSLBuiltin<"lit"> {
  let DetailFunc = "lit_impl";
  let ParamNames = ["NDotL", "NDotH", "M"];
  let Args = [FloatTy, FloatTy, FloatTy];
  let ReturnType = VectorType<FloatTy, 4>;
}

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
