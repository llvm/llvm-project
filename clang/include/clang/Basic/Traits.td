class Trait<string P> {
  string Prefix = P;
  string Spelling = "";
  string StdName = "";
  string KeyFlag = "KEYCXX";
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

// C99 6.4.1: Keywords.  These turn into kw_* tokens.
def SizeOf : UnaryExprOrTypeTrait {
  let Spelling = "sizeof";
  let KeyFlag = "KEYALL";
}

def DataSizeOf : UnaryExprOrTypeTrait {
  let Spelling = "__datasizeof";
}

// C2y
def CountOf : UnaryExprOrTypeTrait {
  let Spelling = "_Countof";
  let KeyFlag = "KEYNOCXX";
}

// C++11 keywords
// alignof and _Alignof return the required ABI alignment
def AlignOf : CXX11UnaryExprOrTypeTrait {
  let Spelling = "alignof";
  let KeyFlag = "KEYC23";
}

// GNU Extensions (in impl-reserved namespace)

// __alignof returns the preferred alignment of a type, the alignment
// clang will attempt to give an object of the type if allowed by ABI.
def PreferredAlignOf : UnaryExprOrTypeTrait {
  let Spelling = "__alignof";
  let KeyFlag = "KEYALL";
}

// __builtin_types_compatible_p is a GNU C extension that we handle like a C++
// type trait.
def TypeCompatible : BinaryTrait {
  let Spelling = "__builtin_types_compatible_p";
  let KeyFlag = "KEYNOCXX";
}

// MS Extensions
def IsInterfaceClass : UnaryTrait {
  let Spelling = "__is_interface_class";
  let KeyFlag = "KEYMS";
}

def IsSealed : UnaryTrait {
  let Spelling = "__is_sealed";
  let KeyFlag = "KEYMS";
}

// MSVC12.0 / VS2013 Type Traits
def IsDestructible : UnaryTrait {
  let Spelling = "__is_destructible";
  let KeyFlag = "KEYALL";
}

def IsTriviallyDestructible : UnaryTrait {
  let Spelling = "__is_trivially_destructible";
}

def IsNothrowDestructible : UnaryTrait {
  let Spelling = "__is_nothrow_destructible";
  let KeyFlag = "KEYALL";
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
  let KeyFlag = "KEYALL";
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
  let KeyFlag = "KEYALL";
}

// OpenCL builtins
def VecStep : UnaryExprOrTypeTrait {
  let Spelling = "vec_step";
  let KeyFlag = "KEYOPENCLC | KEYOPENCLCXX | KEYALTIVEC | KEYZVECTOR";
}

// HLSL Type traits
def IsScalarizedLayoutCompatible : BinaryTrait {
  let Spelling = "__builtin_hlsl_is_scalarized_layout_compatible";
  let KeyFlag = "KEYHLSL";
}

def IsIntangibleType : UnaryTrait {
  let Spelling = "__builtin_hlsl_is_intangible";
  let KeyFlag = "KEYHLSL";
}

def IsTypedResourceElementCompatible : UnaryTrait {
  let Spelling = "__builtin_hlsl_is_typed_resource_element_compatible";
  let KeyFlag = "KEYHLSL";
}

def IsConstantBufferElementCompatible : UnaryTrait {
  let Spelling = "__builtin_hlsl_is_constant_buffer_element_compatible";
  let KeyFlag = "KEYHLSL";
}

// OpenMP Type Traits
def OpenMPRequiredSimdAlign : UnaryExprOrTypeTrait {
  let Spelling = "__builtin_omp_required_simd_align";
  let KeyFlag = "KEYALL";
}

// Clang Extensions.
def VectorElements : UnaryExprOrTypeTrait {
  let Spelling = "__builtin_vectorelements";
  let KeyFlag = "KEYALL";
}
