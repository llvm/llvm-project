// RUN: clang-tblgen -gen-hlsl-alias-intrinsics -I%p/../../include %s | FileCheck %s --check-prefix=ALIAS
// RUN: clang-tblgen -gen-hlsl-inline-intrinsics -I%p/../../include %s | FileCheck %s --check-prefix=INLINE

// Tests for the HLSL intrinsic TableGen backend (HLSLEmitter). Each test def
// exercises a specific feature. Def names are numbered (test_NN_) to control
// TableGen's alphabetical emission order so that CHECK lines read top-to-bottom.

include "clang/Basic/HLSLIntrinsics.td"

//===----------------------------------------------------------------------===//
// Basic alias builtin: scalar + vector overloads, no availability.
// Exercises: Varying arg/return, VaryingScalar, VaryingVecSizes, builtin
// alias emission, multiple regular types.
//===----------------------------------------------------------------------===//

def test_01_basic_alias : HLSLBuiltin<"myfunc", "__builtin_myfunc"> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2, 3];
  let VaryingTypes = [IntTy, FloatTy];
}

// ALIAS-LABEL: // myfunc overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_myfunc)
// ALIAS-NEXT: int myfunc(int);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_myfunc)
// ALIAS-NEXT: int2 myfunc(int2);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_myfunc)
// ALIAS-NEXT: int3 myfunc(int3);
//
// ALIAS-NOT: half myfunc
// ALIAS-NOT: uint myfunc
// ALIAS-NOT: double myfunc
//
// ALIAS: _HLSL_BUILTIN_ALIAS(__builtin_myfunc)
// ALIAS-NEXT: float myfunc(float);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_myfunc)
// ALIAS-NEXT: float2 myfunc(float2);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_myfunc)
// ALIAS-NEXT: float3 myfunc(float3);
//
// ALIAS-NOT: myfunc

//===----------------------------------------------------------------------===//
// VaryingElemType return: return type is scalar even for vector args.
// Exercises: VaryingElemType as ReturnType, two Varying args.
//===----------------------------------------------------------------------===//

def test_02_elem_return : HLSLBuiltin<"dotlike", "__builtin_dotlike"> {
  let Args = [Varying, Varying];
  let ReturnType = VaryingElemType;
  let VaryingScalar = 1;
  let VaryingVecSizes = [3];
  let VaryingTypes = [FloatTy];
}

// ALIAS-LABEL: // dotlike overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_dotlike)
// ALIAS-NEXT: float dotlike(float, float);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_dotlike)
// ALIAS-NEXT: float dotlike(float3, float3);
//
// ALIAS-NOT: dotlike

//===----------------------------------------------------------------------===//
// VaryingShape return: return shape matches arg shape but with a different
// element type.
// Exercises: VaryingShape<BoolTy> as ReturnType.
//===----------------------------------------------------------------------===//

def test_03_shape_return : HLSLBuiltin<"isspecial", "__builtin_isspecial"> {
  let Args = [Varying];
  let ReturnType = VaryingShape<BoolTy>;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2];
  let VaryingTypes = [FloatTy];
}

// ALIAS-LABEL: // isspecial overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_isspecial)
// ALIAS-NEXT: bool isspecial(float);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_isspecial)
// ALIAS-NEXT: bool2 isspecial(float2);
//
// ALIAS-NOT: isspecial

//===----------------------------------------------------------------------===//
// Fixed VectorType args and return: no Varying types at all.
// Exercises: VectorType in Args and ReturnType, single overload.
//===----------------------------------------------------------------------===//

def test_04_fixed_vector : HLSLBuiltin<"fixedfunc", "__builtin_fixedfunc"> {
  let Args = [VectorType<FloatTy, 3>, VectorType<FloatTy, 3>];
  let ReturnType = VectorType<FloatTy, 3>;
}

// ALIAS-LABEL: // fixedfunc overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_fixedfunc)
// ALIAS-NEXT: float3 fixedfunc(float3, float3);
//
// ALIAS-NOT: fixedfunc

//===----------------------------------------------------------------------===//
// Matrix overloads.
// Exercises: VaryingMatDims emission with matrix type names.
//===----------------------------------------------------------------------===//

def test_05_matrix : HLSLBuiltin<"matfunc", "__builtin_matfunc"> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingMatDims = [Mat2x3];
  let VaryingTypes = [FloatTy];
}

// ALIAS-LABEL: // matfunc overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_matfunc)
// ALIAS-NEXT: float2x3 matfunc(float2x3);
//
// ALIAS-NOT: matfunc

//===----------------------------------------------------------------------===//
// Half type: conditionally-16-bit availability.
// Exercises: _HLSL_16BIT_AVAILABILITY for half, no ifdef guard.
//===----------------------------------------------------------------------===//

def test_06_half : HLSLBuiltin<"halffunc", "__builtin_halffunc"> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingTypes = [HalfTy, FloatTy];
}

// ALIAS-LABEL: // halffunc overloads
// ALIAS-NEXT: _HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_halffunc)
// ALIAS-NEXT: half halffunc(half);
//
// ALIAS-NOT: int halffunc
// ALIAS-NOT: double halffunc
//
// ALIAS: _HLSL_BUILTIN_ALIAS(__builtin_halffunc)
// ALIAS-NEXT: float halffunc(float);
//
// ALIAS-NOT: halffunc

//===----------------------------------------------------------------------===//
// 16-bit integer type: ifdef guard + availability.
// Exercises: #ifdef __HLSL_ENABLE_16_BIT guard, _HLSL_AVAILABILITY for
// Is16Bit types, guard closing before non-16-bit types.
//===----------------------------------------------------------------------===//

def test_07_16bit : HLSLBuiltin<"i16func", "__builtin_i16func"> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingTypes = [Int16Ty, IntTy];
}

// ALIAS-LABEL: // i16func overloads
// ALIAS-NEXT: #ifdef __HLSL_ENABLE_16_BIT
// ALIAS-NEXT: _HLSL_AVAILABILITY(shadermodel, 6.2)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_i16func)
// ALIAS-NEXT: int16_t i16func(int16_t);
// ALIAS-NEXT: #endif
//
// ALIAS-NOT: half i16func
// ALIAS-NOT: float i16func
// ALIAS-NOT: double i16func
//
// ALIAS: _HLSL_BUILTIN_ALIAS(__builtin_i16func)
// ALIAS-NEXT: int i16func(int);
//
// ALIAS-NOT: i16func

//===----------------------------------------------------------------------===//
// Explicit availability (>= SM6.2): half and int16_t both use
// _HLSL_AVAILABILITY with the intrinsic's version instead of the 16-bit
// specific macros, since SM >= 6.2 already implies 16-bit support.
// Exercises: Availability field, SM override for conditionally-16-bit and
// 16-bit integer types.
//===----------------------------------------------------------------------===//

def test_08_avail_ge62 : HLSLBuiltin<"availfunc", "__builtin_availfunc"> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingTypes = [HalfTy, Int16Ty, IntTy, FloatTy];
  let Availability = SM6_4;
}

// ALIAS-LABEL: // availfunc overloads
// ALIAS-NEXT: _HLSL_AVAILABILITY(shadermodel, 6.4)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_availfunc)
// ALIAS-NEXT: half availfunc(half);
// ALIAS: #ifdef __HLSL_ENABLE_16_BIT
// ALIAS-NEXT: _HLSL_AVAILABILITY(shadermodel, 6.4)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_availfunc)
// ALIAS-NEXT: int16_t availfunc(int16_t);
// ALIAS-NEXT: #endif
// ALIAS: _HLSL_AVAILABILITY(shadermodel, 6.4)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_availfunc)
// ALIAS-NEXT: int availfunc(int);
// ALIAS: _HLSL_AVAILABILITY(shadermodel, 6.4)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_availfunc)
// ALIAS-NEXT: float availfunc(float);
//
// ALIAS-NOT: availfunc

//===----------------------------------------------------------------------===//
// Explicit availability (< SM6.2) with 16-bit types: int16_t gets the ifdef
// guard and _HLSL_AVAILABILITY(shadermodel, 6.2) regardless of the intrinsic's
// stated availability, because 16-bit types require SM6.2. Half gets a 3-arg
// _HLSL_16BIT_AVAILABILITY so that 16-bit half uses SM6.2 availability while
// the non-16-bit float fallback uses the intrinsic's original availability.
// Exercises: Availability < SM6.2 with Is16Bit and IsConditionally16Bit types.
//===----------------------------------------------------------------------===//

def test_08b_avail_lt62 : HLSLBuiltin<"availlt62", "__builtin_availlt62"> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingTypes = [HalfTy, Int16Ty, IntTy];
  let Availability = SM6_1;
}

// ALIAS-LABEL: // availlt62 overloads
// ALIAS-NEXT: _HLSL_16BIT_AVAILABILITY(shadermodel, 6.2, 6.1)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_availlt62)
// ALIAS-NEXT: half availlt62(half);
// ALIAS: #ifdef __HLSL_ENABLE_16_BIT
// ALIAS-NEXT: _HLSL_AVAILABILITY(shadermodel, 6.2)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_availlt62)
// ALIAS-NEXT: int16_t availlt62(int16_t);
// ALIAS-NEXT: #endif
// ALIAS: _HLSL_AVAILABILITY(shadermodel, 6.1)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_availlt62)
// ALIAS-NEXT: int availlt62(int);
//
// ALIAS-NOT: availlt62

//===----------------------------------------------------------------------===//
// Fixed half VectorType arg: conditionally-16-bit flag from fixed arg.
// Exercises: 16-bit flag propagation through fixed VectorType args.
//===----------------------------------------------------------------------===//

def test_09_fixed_half_vec : HLSLBuiltin<"fixedhalffunc", "__builtin_fixedhalffunc"> {
  let Args = [VectorType<HalfTy, 2>, VectorType<HalfTy, 2>, FloatTy];
  let ReturnType = FloatTy;
}

// ALIAS-LABEL: // fixedhalffunc overloads
// ALIAS-NEXT: _HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_fixedhalffunc)
// ALIAS-NEXT: float fixedhalffunc(half2, half2, float);
//
// ALIAS-NOT: fixedhalffunc

//===----------------------------------------------------------------------===//
// No-arg void function with convergent attribute.
// Exercises: VoidTy return, empty Args, IsConvergent.
//===----------------------------------------------------------------------===//

def test_10_void_convergent : HLSLBuiltin<"barrier", "__builtin_barrier"> {
  let IsConvergent = 1;
}

// ALIAS-LABEL: // barrier overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_barrier)
// ALIAS-NEXT: __attribute__((convergent)) void barrier();
//
// ALIAS-NOT: barrier

//===----------------------------------------------------------------------===//
// Inline body: constexpr function with literal body.
// Exercises: Body field, IsConstexpr, ParamNames, inline backend only.
//===----------------------------------------------------------------------===//

def test_11_inline_body : HLSLBuiltin<"inline_only_func"> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingTypes = [UIntTy];
  let Body = "return V;";
  let ParamNames = ["V"];
  let IsConstexpr = 1;
}

// INLINE-NOT: _HLSL_BUILTIN_ALIAS
// INLINE-LABEL: // inline_only_func overloads
// INLINE-NEXT: constexpr uint inline_only_func(uint V) { return V; }
//
// INLINE-NOT: inline_only_func

//===----------------------------------------------------------------------===//
// Detail function: inline function calling __detail::helper.
// Exercises: DetailFunc, ParamNames, VaryingElemType arg, mixed Varying and
// non-Varying args in inline backend.
//===----------------------------------------------------------------------===//

def test_12_detail_func : HLSLBuiltin<"detailfunc"> {
  let DetailFunc = "detail_impl";
  let Args = [Varying, Varying, VaryingElemType];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2];
  let VaryingTypes = [FloatTy];
  let ParamNames = ["A", "B", "C"];
}

// INLINE-LABEL: // detailfunc overloads
// INLINE-NEXT: inline float detailfunc(float A, float B, float C) {
// INLINE-NEXT:   return __detail::detail_impl(A, B, C);
// INLINE-NEXT: }
// INLINE-NEXT: inline float2 detailfunc(float2 A, float2 B, float C) {
// INLINE-NEXT:   return __detail::detail_impl(A, B, C);
// INLINE-NEXT: }
//
// INLINE-NOT: detailfunc

//===----------------------------------------------------------------------===//
// Doc comment emission.
// Exercises: Doc field producing /// lines with blank-line handling.
//===----------------------------------------------------------------------===//

def test_13_doc : HLSLBuiltin<"docfunc", "__builtin_docfunc"> {
  let Doc = [{
\fn void docfunc()
\brief A documented function.
}];
}

// ALIAS:      /// \fn void docfunc()
// ALIAS-NEXT: /// \brief A documented function.
// ALIAS-NEXT: // docfunc overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_docfunc)
// ALIAS-NEXT: void docfunc();
//
// ALIAS-NOT: docfunc

//===----------------------------------------------------------------------===//
// Canonical type sort order: types emitted in priority order regardless of
// declaration order.
// Exercises: getTypeSortPriority ordering (half before int before float).
//===----------------------------------------------------------------------===//

def test_14_sort_order : HLSLBuiltin<"sortedfunc", "__builtin_sortedfunc"> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingTypes = [FloatTy, IntTy, HalfTy];
}

// ALIAS-LABEL: // sortedfunc overloads
// ALIAS-NEXT: _HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_sortedfunc)
// ALIAS-NEXT: half sortedfunc(half);
//
// ALIAS-NOT: double sortedfunc
// ALIAS-NOT: uint sortedfunc
//
// ALIAS: _HLSL_BUILTIN_ALIAS(__builtin_sortedfunc)
// ALIAS-NEXT: int sortedfunc(int);
// ALIAS: _HLSL_BUILTIN_ALIAS(__builtin_sortedfunc)
// ALIAS-NEXT: float sortedfunc(float);
//
// ALIAS-NOT: sortedfunc

//===----------------------------------------------------------------------===//
// Vector and matrix dimension sort order: vector sizes are emitted ascending
// and matrix dimensions sorted by rows then columns, regardless of declaration
// order.
// Exercises: VaryingVecSizes sorting, VaryingMatDims sorting.
//===----------------------------------------------------------------------===//

def test_15_shape_sort : HLSLBuiltin<"shapesorted", "__builtin_shapesorted"> {
  let Args = [Varying];
  let ReturnType = Varying;
  let VaryingVecSizes = [4, 2, 3];
  let VaryingMatDims = [Mat3x2, Mat2x1, Mat2x3];
  let VaryingTypes = [FloatTy];
}

// ALIAS-LABEL: // shapesorted overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_shapesorted)
// ALIAS-NEXT: float2 shapesorted(float2);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_shapesorted)
// ALIAS-NEXT: float3 shapesorted(float3);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_shapesorted)
// ALIAS-NEXT: float4 shapesorted(float4);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_shapesorted)
// ALIAS-NEXT: float2x1 shapesorted(float2x1);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_shapesorted)
// ALIAS-NEXT: float2x3 shapesorted(float2x3);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_shapesorted)
// ALIAS-NEXT: float3x2 shapesorted(float3x2);
//
// ALIAS-NOT: shapesorted

//===----------------------------------------------------------------------===//
// VaryingElemType as argument: arg is always the scalar element type even when
// the overload shape is a vector.
// Exercises: VaryingElemType in Args position.
//===----------------------------------------------------------------------===//

def test_16_elemtype_arg : HLSLBuiltin<"elemarg", "__builtin_elemarg"> {
  let Args = [Varying, VaryingElemType];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [3];
  let VaryingTypes = [FloatTy];
}

// ALIAS-LABEL: // elemarg overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_elemarg)
// ALIAS-NEXT: float elemarg(float, float);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_elemarg)
// ALIAS-NEXT: float3 elemarg(float3, float);
//
// ALIAS-NOT: elemarg

//===----------------------------------------------------------------------===//
// VaryingShape as argument: arg uses the varying shape but with a fixed element
// type.
// Exercises: VaryingShape<UIntTy> in Args position.
//===----------------------------------------------------------------------===//

def test_17_shape_arg : HLSLBuiltin<"shapearg", "__builtin_shapearg"> {
  let Args = [Varying, VaryingShape<UIntTy>];
  let ReturnType = Varying;
  let VaryingScalar = 1;
  let VaryingVecSizes = [2];
  let VaryingTypes = [FloatTy];
}

// ALIAS-LABEL: // shapearg overloads
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_shapearg)
// ALIAS-NEXT: float shapearg(float, uint);
// ALIAS-NEXT: _HLSL_BUILTIN_ALIAS(__builtin_shapearg)
// ALIAS-NEXT: float2 shapearg(float2, uint2);
//
// ALIAS-NOT: shapearg

//===----------------------------------------------------------------------===//
// Verify no alias intrinsics are generated in the inline header.
//===----------------------------------------------------------------------===//

// INLINE-NOT: myfunc
// INLINE-NOT: dotlike
// INLINE-NOT: isspecial
// INLINE-NOT: fixedfunc
// INLINE-NOT: matfunc
// INLINE-NOT: halffunc
// INLINE-NOT: i16func
// INLINE-NOT: availfunc
// INLINE-NOT: availlt62
// INLINE-NOT: fixedhalffunc
// INLINE-NOT: barrier
// INLINE-NOT: docfunc
// INLINE-NOT: sortedfunc
// INLINE-NOT: shapesorted
// INLINE-NOT: elemarg
// INLINE-NOT: shapearg

//===----------------------------------------------------------------------===//
// Verify no inline intrinsics are generated in the alias header.
//===----------------------------------------------------------------------===//

// ALIAS-NOT: inline_only_func
// ALIAS-NOT: detailfunc

