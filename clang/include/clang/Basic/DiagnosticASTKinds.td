//==--- DiagnosticASTKinds.td - libast diagnostics ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let Component = "AST" in {

// Constant expression diagnostics. These (and their users) belong in Sema.
def note_expr_divide_by_zero : Note<"division by zero">;
def note_constexpr_invalid_cast : Note<
  "%select{reinterpret_cast|dynamic_cast|cast that performs the conversions of"
  " a reinterpret_cast|cast from %1}0 is not allowed in a constant expression"
  "%select{| in C++ standards before C++2a||}0">;
def note_constexpr_invalid_downcast : Note<
  "cannot cast object of dynamic type %0 to type %1">;
def note_constexpr_overflow : Note<
  "value %0 is outside the range of representable values of type %1">;
def note_constexpr_negative_shift : Note<"negative shift count %0">;
def note_constexpr_large_shift : Note<
  "shift count %0 >= width of type %1 (%2 bit%s2)">;
def note_constexpr_lshift_of_negative : Note<"left shift of negative value %0">;
def note_constexpr_lshift_discards : Note<"signed left shift discards bits">;
def note_constexpr_invalid_function : Note<
  "%select{non-constexpr|undefined}0 %select{function|constructor}1 %2 cannot "
  "be used in a constant expression">;
def note_constexpr_invalid_inhctor : Note<
  "constructor inherited from base class %0 cannot be used in a "
  "constant expression; derived class cannot be implicitly initialized">;
def note_constexpr_no_return : Note<
  "control reached end of constexpr function">;
def note_constexpr_virtual_call : Note<
  "cannot evaluate call to virtual function in a constant expression "
  "in C++ standards before C++2a">;
def note_constexpr_pure_virtual_call : Note<
  "pure virtual function %q0 called">;
def note_constexpr_polymorphic_unknown_dynamic_type : Note<
  "%select{|||||virtual function called on|dynamic_cast applied to|"
  "typeid applied to|construction of|destruction of}0 object '%1' "
  "whose dynamic type is not constant">;
def note_constexpr_dynamic_cast_to_reference_failed : Note<
  "reference dynamic_cast failed: %select{"
  "static type %1 of operand is a non-public base class of dynamic type %2|"
  "dynamic type %2 of operand does not have a base class of type %3|"
  "%3 is an ambiguous base class of dynamic type %2 of operand|"
  "%3 is a non-public base class of dynamic type %2 of operand}0">;
def note_constexpr_virtual_base : Note<
  "cannot construct object of type %0 with virtual base class "
  "in a constant expression">;
def note_constexpr_nonliteral : Note<
  "non-literal type %0 cannot be used in a constant expression">;
def note_constexpr_non_global : Note<
  "%select{pointer|reference}0 to %select{|subobject of }1"
  "%select{temporary|%3}2 is not a constant expression">;
def note_constexpr_dynamic_alloc : Note<
  "%select{pointer|reference}0 to %select{|subobject of }1"
  "heap-allocated object is not a constant expression">;
def note_constexpr_uninitialized : Note<
  "%select{|sub}0object of type %1 is not initialized">;
def note_constexpr_subobject_declared_here : Note<
  "subobject declared here">;
def note_constexpr_array_index : Note<"cannot refer to element %0 of "
  "%select{array of %2 element%plural{1:|:s}2|non-array object}1 "
  "in a constant expression">;
def note_constexpr_float_arithmetic : Note<
  "floating point arithmetic produces %select{an infinity|a NaN}0">;
def note_constexpr_pointer_subtraction_not_same_array : Note<
  "subtracted pointers are not elements of the same array">;
def note_constexpr_pointer_subtraction_zero_size : Note<
  "subtraction of pointers to type %0 of zero size">;
def note_constexpr_pointer_comparison_unspecified : Note<
  "comparison has unspecified value">;
def note_constexpr_pointer_comparison_base_classes : Note<
  "comparison of addresses of subobjects of different base classes "
  "has unspecified value">;
def note_constexpr_pointer_comparison_base_field : Note<
  "comparison of address of base class subobject %0 of class %1 to field %2 "
  "has unspecified value">;
def note_constexpr_pointer_comparison_differing_access : Note<
  "comparison of address of fields %0 and %2 of %4 with differing access "
  "specifiers (%1 vs %3) has unspecified value">;
def note_constexpr_compare_virtual_mem_ptr : Note<
  "comparison of pointer to virtual member function %0 has unspecified value">;
def note_constexpr_past_end : Note<
  "dereferenced pointer past the end of %select{|subobject of }0"
  "%select{temporary|%2}1 is not a constant expression">;
def note_constexpr_past_end_subobject : Note<
  "cannot %select{access base class of|access derived class of|access field of|"
  "access array element of|ERROR|"
  "access real component of|access imaginary component of}0 "
  "pointer past the end of object">;
def note_non_null_attribute_failed : Note<
  "null passed to a callee that requires a non-null argument">;
def note_constexpr_null_subobject : Note<
  "cannot %select{access base class of|access derived class of|access field of|"
  "access array element of|perform pointer arithmetic on|"
  "access real component of|"
  "access imaginary component of}0 null pointer">;
def note_constexpr_var_init_non_constant : Note<
  "initializer of %0 is not a constant expression">;
def note_constexpr_typeid_polymorphic : Note<
  "typeid applied to expression of polymorphic type %0 is "
  "not allowed in a constant expression in C++ standards before C++2a">;
def note_constexpr_void_comparison : Note<
  "comparison between unequal pointers to void has unspecified result">;
def note_constexpr_temporary_here : Note<"temporary created here">;
def note_constexpr_dynamic_alloc_here : Note<"heap allocation performed here">;
def note_constexpr_conditional_never_const : Note<
  "both arms of conditional operator are unable to produce a "
  "constant expression">;
def note_constexpr_depth_limit_exceeded : Note<
  "constexpr evaluation exceeded maximum depth of %0 calls">;
def note_constexpr_call_limit_exceeded : Note<
  "constexpr evaluation hit maximum call limit">;
def note_constexpr_step_limit_exceeded : Note<
  "constexpr evaluation hit maximum step limit; possible infinite loop?">;
def note_constexpr_heap_alloc_limit_exceeded : Note<
  "constexpr evaluation hit maximum heap allocation limit">;
def note_constexpr_this : Note<
  "%select{|implicit }0use of 'this' pointer is only allowed within the "
  "evaluation of a call to a 'constexpr' member function">;
def note_constexpr_lifetime_ended : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|construction of|"
  "destruction of}0 %select{temporary|variable}1 whose "
  "%plural{8:storage duration|:lifetime}0 has ended">;
def note_constexpr_access_uninit : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|"
  "construction of subobject of|destruction of}0 "
  "%select{object outside its lifetime|uninitialized object}1 "
  "is not allowed in a constant expression">;
def note_constexpr_use_uninit_reference : Note<
  "use of reference outside its lifetime "
  "is not allowed in a constant expression">;
def note_constexpr_modify_const_type : Note<
  "modification of object of const-qualified type %0 is not allowed "
  "in a constant expression">;
def note_constexpr_access_volatile_type : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "<ERROR>|<ERROR>|<ERROR>|<ERROR>}0 "
  "volatile-qualified type %1 is not allowed in a constant expression">;
def note_constexpr_access_volatile_obj : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "<ERROR>|<ERROR>|<ERROR>|<ERROR>}0 "
  "volatile %select{temporary|object %2|member %2}1 is not allowed in "
  "a constant expression">;
def note_constexpr_volatile_here : Note<
  "volatile %select{temporary created|object declared|member declared}0 here">;
def note_constexpr_access_mutable : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|construction of|"
  "destruction of}0 "
  "mutable member %1 is not allowed in a constant expression">;
def note_constexpr_ltor_non_const_int : Note<
  "read of non-const variable %0 is not allowed in a constant expression">;
def note_constexpr_ltor_non_constexpr : Note<
  "read of non-constexpr variable %0 is not allowed in a constant expression">;
def note_constexpr_ltor_incomplete_type : Note<
  "read of incomplete type %0 is not allowed in a constant expression">;
def note_constexpr_access_null : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|construction of|"
  "destruction of}0 "
  "dereferenced null pointer is not allowed in a constant expression">;
def note_constexpr_access_past_end : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|construction of|"
  "destruction of}0 "
  "dereferenced one-past-the-end pointer is not allowed "
  "in a constant expression">;
def note_constexpr_access_unsized_array : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|construction of|"
  "destruction of}0 "
  "element of array without known bound "
  "is not allowed in a constant expression">;
def note_constexpr_access_inactive_union_member : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|"
  "construction of subobject of|destruction of}0 "
  "member %1 of union with %select{active member %3|no active member}2 "
  "is not allowed in a constant expression">;
def note_constexpr_access_static_temporary : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|reconstruction of|"
  "destruction of}0 temporary "
  "is not allowed in a constant expression outside the expression that "
  "created the temporary">;
def note_constexpr_access_unreadable_object : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|construction of|"
  "destruction of}0 "
  "object '%1' whose value is not known">;
def note_constexpr_access_deleted_object : Note<
  "%select{read of|read of|assignment to|increment of|decrement of|"
  "member call on|dynamic_cast of|typeid applied to|construction of|"
  "destruction of}0 "
  "heap allocated object that has been deleted">;
def note_constexpr_modify_global : Note<
  "a constant expression cannot modify an object that is visible outside "
  "that expression">;
def note_constexpr_stmt_expr_unsupported : Note<
  "this use of statement expressions is not supported in a "
  "constant expression">;
def note_constexpr_calls_suppressed : Note<
  "(skipping %0 call%s0 in backtrace; use -fconstexpr-backtrace-limit=0 to "
  "see all)">;
def note_constexpr_call_here : Note<"in call to '%0'">;
def note_constexpr_inherited_ctor_call_here : Note<
  "in implicit initialization for inherited constructor of %0">;
def note_constexpr_baa_insufficient_alignment : Note<
  "%select{alignment of|offset of the aligned pointer from}0 the base pointee "
  "object (%1 %plural{1:byte|:bytes}1) is %select{less than|not a multiple of}0 the "
  "asserted %2 %plural{1:byte|:bytes}2">;
def note_constexpr_baa_value_insufficient_alignment : Note<
  "value of the aligned pointer (%0) is not a multiple of the asserted %1 "
  "%plural{1:byte|:bytes}1">;
def note_constexpr_destroy_out_of_lifetime : Note<
  "destroying object '%0' whose lifetime has already ended">;
def note_constexpr_unsupported_destruction : Note<
  "non-trivial destruction of type %0 in a constant expression is not supported">;
def note_constexpr_unsupported_tempoarary_nontrivial_dtor : Note<
  "non-trivial destruction of lifetime-extended temporary with type %0 "
  "used in the result of a constant expression is not yet supported">;
def note_constexpr_unsupported_unsized_array : Note<
  "array-to-pointer decay of array member without known bound is not supported">;
def note_constexpr_unsized_array_indexed : Note<
  "indexing of array without known bound is not allowed "
  "in a constant expression">;
def note_constexpr_memcpy_null : Note<
  "%select{source|destination}2 of "
  "'%select{%select{memcpy|wmemcpy}1|%select{memmove|wmemmove}1}0' "
  "is %3">;
def note_constexpr_memcpy_type_pun : Note<
  "cannot constant evaluate '%select{memcpy|memmove}0' from object of "
  "type %1 to object of type %2">;
def note_constexpr_memcpy_nontrivial : Note<
  "cannot constant evaluate '%select{memcpy|memmove}0' between objects of "
  "non-trivially-copyable type %1">;
def note_constexpr_memcpy_incomplete_type : Note<
  "cannot constant evaluate '%select{memcpy|memmove}0' between objects of "
  "incomplete type %1">;
def note_constexpr_memcpy_overlap : Note<
  "'%select{memcpy|wmemcpy}0' between overlapping memory regions">;
def note_constexpr_memcpy_unsupported : Note<
  "'%select{%select{memcpy|wmemcpy}1|%select{memmove|wmemmove}1}0' "
  "not supported: %select{"
  "size to copy (%4) is not a multiple of size of element type %3 (%5)|"
  "source is not a contiguous array of at least %4 elements of type %3|"
  "destination is not a contiguous array of at least %4 elements of type %3}2">;
def note_constexpr_bit_cast_unsupported_type : Note<
  "constexpr bit_cast involving type %0 is not yet supported">;
def note_constexpr_bit_cast_unsupported_bitfield : Note<
  "constexpr bit_cast involving bit-field is not yet supported">;
def note_constexpr_bit_cast_invalid_type : Note<
  "bit_cast %select{from|to}0 a %select{|type with a }1"
  "%select{union|pointer|member pointer|volatile|reference}2 "
  "%select{type|member}1 is not allowed in a constant expression">;
def note_constexpr_bit_cast_invalid_subtype : Note<
  "invalid type %0 is a %select{member|base}1 of %2">;
def note_constexpr_bit_cast_indet_dest : Note<
  "indeterminate value can only initialize an object of type 'unsigned char'"
  "%select{, 'char',|}1 or 'std::byte'; %0 is invalid">;
def note_constexpr_pseudo_destructor : Note<
  "pseudo-destructor call is not permitted in constant expressions "
  "until C++20">;
def note_constexpr_construct_complex_elem : Note<
  "construction of individual component of complex number is not yet supported "
  "in constant expressions">;
def note_constexpr_destroy_complex_elem : Note<
  "destruction of individual component of complex number is not yet supported "
  "in constant expressions">;
def note_constexpr_new : Note<
  "dynamic memory allocation is not permitted in constant expressions "
  "until C++20">;
def note_constexpr_new_non_replaceable : Note<
  "call to %select{placement|class-specific}0 %1">;
def note_constexpr_new_placement : Note<
  "this placement new expression is not yet supported in constant expressions">;
def note_constexpr_placement_new_wrong_type : Note<
  "placement new would change type of storage from %0 to %1">;
def note_constexpr_new_negative : Note<
  "cannot allocate array; evaluated array bound %0 is negative">;
def note_constexpr_new_too_large : Note<
  "cannot allocate array; evaluated array bound %0 is too large">;
def note_constexpr_new_too_small : Note<
  "cannot allocate array; evaluated array bound %0 is too small to hold "
  "%1 explicitly initialized elements">;
def note_constexpr_new_untyped : Note<
  "cannot allocate untyped memory in a constant expression; "
  "use 'std::allocator<T>::allocate' to allocate memory of type 'T'">;
def note_constexpr_new_not_complete_object_type : Note<
  "cannot allocate memory of %select{incomplete|function}0 type %1">;
def note_constexpr_operator_new_bad_size : Note<
  "allocated size %0 is not a multiple of size %1 of element type %2">;
def note_constexpr_delete_not_heap_alloc : Note<
  "delete of pointer '%0' that does not point to a heap-allocated object">;
def note_constexpr_double_delete : Note<
  "delete of pointer that has already been deleted">;
def note_constexpr_double_destroy : Note<
  "destruction of object that is already being destroyed">;
def note_constexpr_new_delete_mismatch : Note<
  "%plural{2:'delete' used to delete pointer to object "
  "allocated with 'std::allocator<...>::allocate'|"
  ":%select{non-array delete|array delete|'std::allocator<...>::deallocate'}0 "
  "used to delete pointer to "
  "%select{array object of type %2|non-array object of type %2|"
  "object allocated with 'new'}0}1">;
def note_constexpr_delete_subobject : Note<
  "delete of pointer%select{ to subobject|}1 '%0' "
  "%select{|that does not point to complete object}1">;
def note_constexpr_delete_base_nonvirt_dtor : Note<
  "delete of object with dynamic type %1 through pointer to "
  "base class type %0 with non-virtual destructor">;
def note_constexpr_memory_leak : Note<
  "allocation performed here was not deallocated"
  "%plural{0:|: (along with %0 other memory leak%s0)}0">;
def err_experimental_clang_interp_failed : Error<
  "the experimental clang interpreter failed to evaluate an expression">;

def warn_integer_constant_overflow : Warning<
  "overflow in expression; result is %0 with type %1">,
  InGroup<DiagGroup<"integer-overflow">>;

// This is a temporary diagnostic, and shall be removed once our
// implementation is complete, and like the preceding constexpr notes belongs
// in Sema.
def note_unimplemented_constexpr_lambda_feature_ast : Note<
    "unimplemented constexpr lambda feature: %0 (coming soon!)">;

def warn_is_constant_evaluated_always_true_constexpr : Warning<
  "'%0' will always evaluate to 'true' in a manifestly constant-evaluated expression">,
  InGroup<DiagGroup<"constant-evaluated">>;

// inline asm related.
let CategoryName = "Inline Assembly Issue" in {
  def err_asm_invalid_escape : Error<
    "invalid %% escape in inline assembly string">;
  def err_asm_unknown_symbolic_operand_name : Error<
    "unknown symbolic operand name in inline assembly string">;

  def err_asm_unterminated_symbolic_operand_name : Error<
    "unterminated symbolic operand name in inline assembly string">;
  def err_asm_empty_symbolic_operand_name : Error<
    "empty symbolic operand name in inline assembly string">;
  def err_asm_invalid_operand_number : Error<
    "invalid operand number in inline asm string">;
}

// vtable related.
let CategoryName = "VTable ABI Issue" in {
  def err_vftable_ambiguous_component : Error<
    "ambiguous vftable component for %0 introduced via covariant thunks; "
    "this is an inherent limitation of the ABI">;
  def note_covariant_thunk : Note<
    "covariant thunk required by %0">;
}

// Importing ASTs
def err_odr_variable_type_inconsistent : Error<
  "external variable %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">;
def warn_odr_variable_type_inconsistent : Warning<
  "external variable %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">,
  InGroup<ODR>;
def err_odr_variable_multiple_def : Error<
  "external variable %0 defined in multiple translation units">;
def warn_odr_variable_multiple_def : Warning<
  "external variable %0 defined in multiple translation units">,
  InGroup<ODR>;
def note_odr_value_here : Note<"declared here with type %0">;
def err_odr_function_type_inconsistent : Error<
  "external function %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">;
def warn_odr_function_type_inconsistent : Warning<
  "external function %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">,
  InGroup<ODR>;
def err_odr_tag_type_inconsistent
    : Error<"type %0 has incompatible definitions in different translation "
            "units">;
def warn_odr_tag_type_inconsistent
    : Warning<"type %0 has incompatible definitions in different translation "
              "units">,
      InGroup<ODR>;
def note_odr_tag_kind_here: Note<
  "%0 is a %select{struct|interface|union|class|enum}1 here">;
def note_odr_field : Note<"field %0 has type %1 here">;
def note_odr_field_name : Note<"field has name %0 here">;
def note_odr_missing_field : Note<"no corresponding field here">;
def note_odr_bit_field : Note<"bit-field %0 with type %1 and length %2 here">;
def note_odr_not_bit_field : Note<"field %0 is not a bit-field">;
def note_odr_base : Note<"class has base type %0">;
def note_odr_virtual_base : Note<
  "%select{non-virtual|virtual}0 derivation here">;
def note_odr_missing_base : Note<"no corresponding base class here">;
def note_odr_number_of_bases : Note<
  "class has %0 base %plural{1:class|:classes}0">;
def note_odr_enumerator : Note<"enumerator %0 with value %1 here">;
def note_odr_missing_enumerator : Note<"no corresponding enumerator here">;
def err_odr_field_type_inconsistent : Error<
  "field %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">;
def warn_odr_field_type_inconsistent : Warning<
  "field %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">,
  InGroup<ODR>;

// Importing Objective-C ASTs
def err_odr_ivar_type_inconsistent : Error<
  "instance variable %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">;
def warn_odr_ivar_type_inconsistent : Warning<
  "instance variable %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">,
  InGroup<ODR>;
def err_odr_objc_superclass_inconsistent : Error<
  "class %0 has incompatible superclasses">;
def warn_odr_objc_superclass_inconsistent : Warning<
  "class %0 has incompatible superclasses">,
  InGroup<ODR>;
def note_odr_objc_superclass : Note<"inherits from superclass %0 here">;
def note_odr_objc_missing_superclass : Note<"no corresponding superclass here">;
def err_odr_objc_method_result_type_inconsistent : Error<
  "%select{class|instance}0 method %1 has incompatible result types in "
  "different translation units (%2 vs. %3)">;
def warn_odr_objc_method_result_type_inconsistent : Warning<
  "%select{class|instance}0 method %1 has incompatible result types in "
  "different translation units (%2 vs. %3)">,
  InGroup<ODR>;
def err_odr_objc_method_num_params_inconsistent : Error<
  "%select{class|instance}0 method %1 has a different number of parameters in "
  "different translation units (%2 vs. %3)">;
def warn_odr_objc_method_num_params_inconsistent : Warning<
  "%select{class|instance}0 method %1 has a different number of parameters in "
  "different translation units (%2 vs. %3)">,
  InGroup<ODR>;
def err_odr_objc_method_param_type_inconsistent : Error<
  "%select{class|instance}0 method %1 has a parameter with a different types "
  "in different translation units (%2 vs. %3)">;
def warn_odr_objc_method_param_type_inconsistent : Warning<
  "%select{class|instance}0 method %1 has a parameter with a different types "
  "in different translation units (%2 vs. %3)">,
  InGroup<ODR>;
def err_odr_objc_method_variadic_inconsistent : Error<
  "%select{class|instance}0 method %1 is variadic in one translation unit "
  "and not variadic in another">;
def warn_odr_objc_method_variadic_inconsistent : Warning<
  "%select{class|instance}0 method %1 is variadic in one translation unit "
  "and not variadic in another">,
  InGroup<ODR>;
def note_odr_objc_method_here : Note<
  "%select{class|instance}0 method %1 also declared here">;
def err_odr_objc_property_type_inconsistent : Error<
  "property %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">;
def warn_odr_objc_property_type_inconsistent : Warning<
  "property %0 declared with incompatible types in different "
  "translation units (%1 vs. %2)">,
  InGroup<ODR>;
def err_odr_objc_property_impl_kind_inconsistent : Error<
  "property %0 is implemented with %select{@synthesize|@dynamic}1 in one "
  "translation but %select{@dynamic|@synthesize}1 in another translation unit">;
def warn_odr_objc_property_impl_kind_inconsistent : Warning<
  "property %0 is implemented with %select{@synthesize|@dynamic}1 in one "
  "translation but %select{@dynamic|@synthesize}1 in another translation unit">,
  InGroup<ODR>;
def note_odr_objc_property_impl_kind : Note<
  "property %0 is implemented with %select{@synthesize|@dynamic}1 here">;
def err_odr_objc_synthesize_ivar_inconsistent : Error<
  "property %0 is synthesized to different ivars in different translation "
  "units (%1 vs. %2)">;
def warn_odr_objc_synthesize_ivar_inconsistent : Warning<
  "property %0 is synthesized to different ivars in different translation "
  "units (%1 vs. %2)">,
  InGroup<ODR>;
def note_odr_objc_synthesize_ivar_here : Note<
  "property is synthesized to ivar %0 here">;

// Importing C++ ASTs
def note_odr_friend : Note<"friend declared here">;
def note_odr_missing_friend : Note<"no corresponding friend here">;
def err_odr_different_num_template_parameters : Error<
  "template parameter lists have a different number of parameters (%0 vs %1)">;
def warn_odr_different_num_template_parameters : Warning<
  "template parameter lists have a different number of parameters (%0 vs %1)">,
  InGroup<ODR>;
def note_odr_template_parameter_list : Note<
  "template parameter list also declared here">;
def err_odr_different_template_parameter_kind : Error<
  "template parameter has different kinds in different translation units">;
def warn_odr_different_template_parameter_kind : Warning<
  "template parameter has different kinds in different translation units">,
  InGroup<ODR>;
def note_odr_template_parameter_here : Note<
  "template parameter declared here">;
def err_odr_parameter_pack_non_pack : Error<
  "parameter kind mismatch; parameter is %select{not a|a}0 parameter pack">;
def warn_odr_parameter_pack_non_pack : Warning<
  "parameter kind mismatch; parameter is %select{not a|a}0 parameter pack">,
  InGroup<ODR>;
def note_odr_parameter_pack_non_pack : Note<
  "%select{parameter|parameter pack}0 declared here">;
def err_odr_non_type_parameter_type_inconsistent : Error<
  "non-type template parameter declared with incompatible types in different "
  "translation units (%0 vs. %1)">;
def warn_odr_non_type_parameter_type_inconsistent : Warning<
  "non-type template parameter declared with incompatible types in different "
  "translation units (%0 vs. %1)">,
  InGroup<ODR>;
def err_unsupported_ast_node: Error<"cannot import unsupported AST node %0">;

def remark_sanitize_address_insert_extra_padding_accepted : Remark<
    "-fsanitize-address-field-padding applied to %0">, ShowInSystemHeader,
    InGroup<SanitizeAddressRemarks>;
def remark_sanitize_address_insert_extra_padding_rejected : Remark<
    "-fsanitize-address-field-padding ignored for %0 because it "
    "%select{is not C++|is packed|is a union|is trivially copyable|"
    "has trivial destructor|is standard layout|is in a blacklisted file|"
    "is blacklisted}1">, ShowInSystemHeader,
    InGroup<SanitizeAddressRemarks>;

def warn_npot_ms_struct : Warning<
  "ms_struct may not produce Microsoft-compatible layouts with fundamental "
  "data types with sizes that aren't a power of two">,
  DefaultError, InGroup<IncompatibleMSStruct>;

// -Wpadded, -Wpacked
def warn_padded_struct_field : Warning<
  "padding %select{struct|interface|class}0 %1 with %2 "
  "%select{byte|bit}3%s2 to align %4">,
  InGroup<Padded>, DefaultIgnore;
def warn_padded_struct_anon_field : Warning<
  "padding %select{struct|interface|class}0 %1 with %2 "
  "%select{byte|bit}3%s2 to align anonymous bit-field">,
  InGroup<Padded>, DefaultIgnore;
def warn_padded_struct_size : Warning<
  "padding size of %0 with %1 %select{byte|bit}2%s1 to alignment boundary">,
  InGroup<Padded>, DefaultIgnore;
def warn_unnecessary_packed : Warning<
  "packed attribute is unnecessary for %0">, InGroup<Packed>, DefaultIgnore;
}
