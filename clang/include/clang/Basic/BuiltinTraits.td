//==--- Traits.td - expression and type traits ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class TokenKey<int V> {
  int Value = V;
}

def KEYC99 : TokenKey<0x1>;
def KEYCXX : TokenKey<0x2>;
def KEYCXX11 : TokenKey<0x4>;
def KEYGNU : TokenKey<0x8>;
def KEYMS : TokenKey<0x10>;
def BOOLSUPPORT : TokenKey<0x20>;
def KEYALTIVEC : TokenKey<0x40>;
def KEYNOCXX : TokenKey<0x80>;
def KEYBORLAND : TokenKey<0x100>;
def KEYOPENCLC : TokenKey<0x200>;
def KEYC23 : TokenKey<0x400>;
def KEYNOMS18 : TokenKey<0x800>;
def KEYNOOPENCL : TokenKey<0x1000>;
def WCHARSUPPORT : TokenKey<0x2000>;
def HALFSUPPORT : TokenKey<0x4000>;
def CHAR8SUPPORT : TokenKey<0x8000>;
def KEYOBJC : TokenKey<0x10000>;
def KEYZVECTOR : TokenKey<0x20000>;
def KEYCOROUTINES : TokenKey<0x40000>;
def KEYMODULES : TokenKey<0x80000>;
def KEYCXX20 : TokenKey<0x100000>;
def KEYOPENCLCXX : TokenKey<0x200000>;
def KEYMSCOMPAT : TokenKey<0x400000>;
def KEYSYCL : TokenKey<0x800000>;
def KEYCUDA : TokenKey<0x1000000>;
def KEYZOS : TokenKey<0x2000000>;
def KEYNOZOS : TokenKey<0x4000000>;
def KEYHLSL : TokenKey<0x8000000>;
def KEYFIXEDPOINT : TokenKey<0x10000000>;
def KEYDEFERTS : TokenKey<0x20000000>;
def KEYNOHLSL : TokenKey<0x40000000>;

def KEYMAX : TokenKey<KEYNOHLSL.Value>;
def KEYALLCXX : TokenKey<!or(KEYCXX.Value, KEYCXX11.Value, KEYCXX20.Value)>;
def KEYALL : TokenKey<!and(!or(KEYMAX.Value, !sub(KEYMAX.Value, 1)),
                           !xor(KEYNOMS18.Value, -1),
                           !xor(KEYNOOPENCL.Value, -1),
                           !xor(KEYNOZOS.Value, -1),
                           !xor(KEYNOHLSL.Value, -1))>;
class Trait<string P> {
  string Prefix = P;
  string Spelling = "";
  string StdName = "";
  list<TokenKey> KeyFlags = [KEYCXX];
}
class TypeTrait<string P> : Trait<P>;

class UnaryTrait : TypeTrait<"UTT">;
class BinaryTrait : TypeTrait<"BTT">;
class VariadicTrait : TypeTrait<"TT">;
class ArrayTrait : Trait<"ATT">;
class ExpressionTrait : Trait<"ET">;
class UnaryExprOrTypeTrait : Trait<"UETT">;
class CXX11UnaryExprOrTypeTrait : Trait<"UETT">;
class TransformTypeTrait : Trait<"TST">;

class Alias<Trait P> {
  Trait Primary = P;
  string Spelling = "";
  list<TokenKey> KeyFlags = [KEYCXX];
}

// C99 6.4.1: Keywords.  These turn into kw_* tokens.
def SizeOf : UnaryExprOrTypeTrait {
  let Spelling = "sizeof";
  let KeyFlags = [KEYALL];
}

def DataSizeOf : UnaryExprOrTypeTrait {
  let Spelling = "__datasizeof";
}

// C2y
def CountOf : UnaryExprOrTypeTrait {
  let Spelling = "_Countof";
  let KeyFlags = [KEYNOCXX];
}

// C++11 keywords
// alignof and _Alignof return the required ABI alignment
def AlignOf : CXX11UnaryExprOrTypeTrait {
  let Spelling = "alignof";
  let KeyFlags = [KEYC23];
}

// GNU Extensions (in impl-reserved namespace)

// __alignof returns the preferred alignment of a type, the alignment
// clang will attempt to give an object of the type if allowed by ABI.
def PreferredAlignOf : UnaryExprOrTypeTrait {
  let Spelling = "__alignof";
  let KeyFlags = [KEYALL];
}

def PreferredAlignOfAlt : Alias<PreferredAlignOf> {
  let Spelling = "__alignof__";
  let KeyFlags = [KEYALL];
}

def BuiltinAlignOf : Alias<PreferredAlignOf> {
  let Spelling = "__builtin_alignof";
  let KeyFlags = [KEYMS];
}

def MSAlignOf : Alias<PreferredAlignOf> {
  let Spelling = "_alignof";
  let KeyFlags = [KEYMS];
}

// __builtin_types_compatible_p is a GNU C extension that we handle like a C++
// type trait.
def TypeCompatible : BinaryTrait {
  let Spelling = "__builtin_types_compatible_p";
  let KeyFlags = [KEYNOCXX];
}

// MS Extensions
def IsInterfaceClass : UnaryTrait {
  let Spelling = "__is_interface_class";
  let KeyFlags = [KEYMS];
}

def IsSealed : UnaryTrait {
  let Spelling = "__is_sealed";
  let KeyFlags = [KEYMS];
}

// MSVC12.0 / VS2013 Type Traits
def IsDestructible : UnaryTrait {
  let Spelling = "__is_destructible";
  let KeyFlags = [KEYALL];
}

def IsTriviallyDestructible : UnaryTrait {
  let Spelling = "__is_trivially_destructible";
}

def IsNothrowDestructible : UnaryTrait {
  let Spelling = "__is_nothrow_destructible";
  let KeyFlags = [KEYALL];
}

def IsNothrowAssignable : BinaryTrait {
  let Spelling = "__is_nothrow_assignable";
}

def IsConstructible : VariadicTrait {
  let Spelling = "__is_constructible";
  let StdName = "is_constructible";
}

def IsNothrowConstructible : VariadicTrait {
  let Spelling = "__is_nothrow_constructible";
}

// MSVC14.0 / VS2015 Type Traits
def IsAssignable : BinaryTrait {
  let Spelling = "__is_assignable";
  let StdName = "is_assignable";
}

// MSVC Type Traits of unknown vintage
def HasNothrowMoveAssign : UnaryTrait {
  let Spelling = "__has_nothrow_move_assign";
}

def HasTrivialMoveAssign : UnaryTrait {
  let Spelling = "__has_trivial_move_assign";
}

def HasTrivialMoveConstructor : UnaryTrait {
  let Spelling = "__has_trivial_move_constructor";
}

// GNU and MS Type Traits
def IsImplicitLifetime : UnaryTrait {
  let Spelling = "__builtin_is_implicit_lifetime";
}

def IsVirtualBaseOf : BinaryTrait {
  let Spelling = "__builtin_is_virtual_base_of";
}

def HasNothrowAssign : UnaryTrait {
  let Spelling = "__has_nothrow_assign";
}

def HasNothrowCopy : UnaryTrait {
  let Spelling = "__has_nothrow_copy";
}

def HasNothrowConstructor : UnaryTrait {
  let Spelling = "__has_nothrow_constructor";
}

def HasTrivialAssign : UnaryTrait {
  let Spelling = "__has_trivial_assign";
}

def HasTrivialCopy : UnaryTrait {
  let Spelling = "__has_trivial_copy";
}

def HasTrivialDefaultConstructor : UnaryTrait {
  let Spelling = "__has_trivial_constructor";
}

def HasTrivialDestructor : UnaryTrait {
  let Spelling = "__has_trivial_destructor";
}

def HasVirtualDestructor : UnaryTrait {
  let Spelling = "__has_virtual_destructor";
}

def IsAbstract : UnaryTrait {
  let Spelling = "__is_abstract";
  let StdName = "is_abstract";
}

def IsAggregate : UnaryTrait {
  let Spelling = "__is_aggregate";
  let StdName = "is_aggregate";
}

def IsBaseOf : BinaryTrait {
  let Spelling = "__is_base_of";
}

def IsClass : UnaryTrait {
  let Spelling = "__is_class";
}

def IsConvertibleTo : BinaryTrait {
  let Spelling = "__is_convertible_to";
}

def IsEmpty : UnaryTrait {
  let Spelling = "__is_empty";
  let StdName = "is_empty";
}

def IsEnum : UnaryTrait {
  let Spelling = "__is_enum";
}

def IsFinal : UnaryTrait {
  let Spelling = "__is_final";
  let StdName = "is_final";
}

def IsLiteral : UnaryTrait {
  let Spelling = "__is_literal";
}

// Name for GCC 4.6 compatibility - people have already written libraries using
// this name unfortunately.
def IsLiteralType : Alias<IsLiteral> {
  let Spelling = "__is_literal_type";
}

def IsPOD : UnaryTrait {
  let Spelling = "__is_pod";
}

def IsPolymorphic : UnaryTrait {
  let Spelling = "__is_polymorphic";
}

def IsStandardLayout : UnaryTrait {
  let Spelling = "__is_standard_layout";
  let StdName = "is_standard_layout";
}

def IsTrivial : UnaryTrait {
  let Spelling = "__is_trivial";
}

def IsTriviallyAssignable : BinaryTrait {
  let Spelling = "__is_trivially_assignable";
}

def IsTriviallyConstructible : VariadicTrait {
  let Spelling = "__is_trivially_constructible";
}

def IsTriviallyCopyable : UnaryTrait {
  let Spelling = "__is_trivially_copyable";
  let StdName = "is_trivially_copyable";
}

def IsUnion : UnaryTrait {
  let Spelling = "__is_union";
}

def HasUniqueObjectRepresentations : UnaryTrait {
  let Spelling = "__has_unique_object_representations";
}

def IsLayoutCompatible : BinaryTrait {
  let Spelling = "__is_layout_compatible";
}

def IsPointerInterconvertibleBaseOf : BinaryTrait {
  let Spelling = "__is_pointer_interconvertible_base_of";
}

def AddLvalueReference : TransformTypeTrait {
  let Spelling = "__add_lvalue_reference";
  let StdName = "add_lvalue_reference";
}

def AddPointer : TransformTypeTrait {
  let Spelling = "__add_pointer";
  let StdName = "add_pointer";
}

def AddRvalueReference : TransformTypeTrait {
  let Spelling = "__add_rvalue_reference";
  let StdName = "add_rvalue_reference";
}

def Decay : TransformTypeTrait {
  let Spelling = "__decay";
  let StdName = "decay";
}

def MakeSigned : TransformTypeTrait {
  let Spelling = "__make_signed";
  let StdName = "make_signed";
}

def MakeUnsigned : TransformTypeTrait {
  let Spelling = "__make_unsigned";
  let StdName = "make_unsigned";
}

def RemoveAllExtents : TransformTypeTrait {
  let Spelling = "__remove_all_extents";
  let StdName = "remove_all_extents";
}

def RemoveConst : TransformTypeTrait {
  let Spelling = "__remove_const";
  let StdName = "remove_const";
}

def RemoveCV : TransformTypeTrait {
  let Spelling = "__remove_cv";
  let StdName = "remove_cv";
}

def RemoveCVRef : TransformTypeTrait {
  let Spelling = "__remove_cvref";
  let StdName = "remove_cvref";
}

def RemoveExtent : TransformTypeTrait {
  let Spelling = "__remove_extent";
  let StdName = "remove_extent";
}

def RemovePointer : TransformTypeTrait {
  let Spelling = "__remove_pointer";
  let StdName = "remove_pointer";
}

def RemoveReference : TransformTypeTrait {
  let Spelling = "__remove_reference_t";
  let StdName = "remove_reference_t";
}

def RemoveRestrict : TransformTypeTrait {
  let Spelling = "__remove_restrict";
  let StdName = "remove_restrict";
}

def RemoveVolatile : TransformTypeTrait {
  let Spelling = "__remove_volatile";
  let StdName = "remove_volatile";
}

def EnumUnderlyingType : TransformTypeTrait {
  let Spelling = "__underlying_type";
  let StdName = "underlying_type";
}

// Clang-only C++ Type Traits
def IsTriviallyEqualityComparable : UnaryTrait {
  let Spelling = "__is_trivially_equality_comparable";
}

def IsBoundedArray : UnaryTrait {
  let Spelling = "__is_bounded_array";
}

def IsUnboundedArray : UnaryTrait {
  let Spelling = "__is_unbounded_array";
}

def IsScopedEnum : UnaryTrait {
  let Spelling = "__is_scoped_enum";
}

def CanPassInRegs : UnaryTrait {
  let Spelling = "__can_pass_in_regs";
}

def ReferenceBindsToTemporary : BinaryTrait {
  let Spelling = "__reference_binds_to_temporary";
}

def ReferenceConstructsFromTemporary : BinaryTrait {
  let Spelling = "__reference_constructs_from_temporary";
}

def ReferenceConvertsFromTemporary : BinaryTrait {
  let Spelling = "__reference_converts_from_temporary";
}

def LtSynthesizesFromSpaceship : BinaryTrait {
  let Spelling = "__builtin_lt_synthesizes_from_spaceship";
}

def LeSynthesizesFromSpaceship : BinaryTrait {
  let Spelling = "__builtin_le_synthesizes_from_spaceship";
}

def GtSynthesizesFromSpaceship : BinaryTrait {
  let Spelling = "__builtin_gt_synthesizes_from_spaceship";
}

def GeSynthesizesFromSpaceship : BinaryTrait {
  let Spelling = "__builtin_ge_synthesizes_from_spaceship";
}

// IsDeducible is only used internally by clang for CTAD implementation and
// is not exposed to users.
def IsDeducible : BinaryTrait {}

// __is_trivially_relocatable is deprecated
def IsCppTriviallyRelocatable : UnaryTrait {
  let Spelling = "__builtin_is_cpp_trivially_relocatable";
  let StdName = "is_trivially_relocatable";
}

def IsTriviallyRelocatable : UnaryTrait {
  let Spelling = "__is_trivially_relocatable";
}

def IsBitwiseCloneable : UnaryTrait {
  let Spelling = "__is_bitwise_cloneable";
  let KeyFlags = [KEYALL];
}

def StructuredBindingSize : UnaryTrait {
  let Spelling = "__builtin_structured_binding_size";
}

// Embarcadero Expression Traits
def IsLValueExpr : ExpressionTrait {
  let Spelling = "__is_lvalue_expr";
}

def IsRValueExpr : ExpressionTrait {
  let Spelling = "__is_rvalue_expr";
}

// Embarcadero Unary Type Traits
def IsArithmetic : UnaryTrait {
  let Spelling = "__is_arithmetic";
}

def IsFloatingPoint : UnaryTrait {
  let Spelling = "__is_floating_point";
}

def IsIntegral : UnaryTrait {
  let Spelling = "__is_integral";
}

def IsCompleteType : UnaryTrait {
  let Spelling = "__is_complete_type";
}

def IsVoid : UnaryTrait {
  let Spelling = "__is_void";
}

def IsArray : UnaryTrait {
  let Spelling = "__is_array";
}

def IsFunction : UnaryTrait {
  let Spelling = "__is_function";
}

def IsReference : UnaryTrait {
  let Spelling = "__is_reference";
}

def IsLvalueReference : UnaryTrait {
  let Spelling = "__is_lvalue_reference";
}

def IsRvalueReference : UnaryTrait {
  let Spelling = "__is_rvalue_reference";
}

def IsFundamental : UnaryTrait {
  let Spelling = "__is_fundamental";
}

def IsObject : UnaryTrait {
  let Spelling = "__is_object";
}

def IsScalar : UnaryTrait {
  let Spelling = "__is_scalar";
}

def IsCompound : UnaryTrait {
  let Spelling = "__is_compound";
}

def IsPointer : UnaryTrait {
  let Spelling = "__is_pointer";
}

def IsMemberObjectPointer : UnaryTrait {
  let Spelling = "__is_member_object_pointer";
}

def IsMemberFunctionPointer : UnaryTrait {
  let Spelling = "__is_member_function_pointer";
}

def IsMemberPointer : UnaryTrait {
  let Spelling = "__is_member_pointer";
}

def IsConst : UnaryTrait {
  let Spelling = "__is_const";
}

def IsVolatile : UnaryTrait {
  let Spelling = "__is_volatile";
}

def IsSigned : UnaryTrait {
  let Spelling = "__is_signed";
}

def IsUnsigned : UnaryTrait {
  let Spelling = "__is_unsigned";
}

// Embarcadero Binary Type Traits
def IsSame : BinaryTrait {
  let Spelling = "__is_same";
}

// Name for GCC 6 compatibility
def IsSameAs : Alias<IsSame> {
  let Spelling = "__is_same_as";
}

def IsConvertible : BinaryTrait {
  let Spelling = "__is_convertible";
}

def IsNothrowConvertible : BinaryTrait {
  let Spelling = "__is_nothrow_convertible";
}

def ArrayRank : ArrayTrait {
  let Spelling = "__array_rank";
}

def ArrayExtent : ArrayTrait {
  let Spelling = "__array_extent";
}

// Apple Extension.
def PtrAuthTypeDiscriminator : UnaryExprOrTypeTrait {
  let Spelling = "__builtin_ptrauth_type_discriminator";
  let KeyFlags = [KEYALL];
}

// OpenCL builtins
def VecStep : UnaryExprOrTypeTrait {
  let Spelling = "vec_step";
  let KeyFlags = [KEYOPENCLC, KEYOPENCLCXX, KEYALTIVEC, KEYZVECTOR];
}

// HLSL Type traits
def IsScalarizedLayoutCompatible : BinaryTrait {
  let Spelling = "__builtin_hlsl_is_scalarized_layout_compatible";
  let KeyFlags = [KEYHLSL];
}

def IsIntangibleType : UnaryTrait {
  let Spelling = "__builtin_hlsl_is_intangible";
  let KeyFlags = [KEYHLSL];
}

def IsTypedResourceElementCompatible : UnaryTrait {
  let Spelling = "__builtin_hlsl_is_typed_resource_element_compatible";
  let KeyFlags = [KEYHLSL];
}

def IsConstantBufferElementCompatible : UnaryTrait {
  let Spelling = "__builtin_hlsl_is_constant_buffer_element_compatible";
  let KeyFlags = [KEYHLSL];
}

// OpenMP Type Traits
def OpenMPRequiredSimdAlign : UnaryExprOrTypeTrait {
  let Spelling = "__builtin_omp_required_simd_align";
  let KeyFlags = [KEYALL];
}

// Clang Extensions.
def VectorElements : UnaryExprOrTypeTrait {
  let Spelling = "__builtin_vectorelements";
  let KeyFlags = [KEYALL];
}
