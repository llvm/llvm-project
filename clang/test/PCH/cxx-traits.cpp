// Test this without pch.
// RUN: %clang_cc1 -fms-extensions -include %S/cxx-traits.h -std=c++11 -fsyntax-only -verify %s

// RUN: %clang_cc1 -fms-extensions -x c++-header -std=c++11 -emit-pch -o %t %S/cxx-traits.h
// RUN: %clang_cc1 -fms-extensions -std=c++11 -include-pch %t -DPCH -fsyntax-only -verify %s

#ifdef PCH
// expected-no-diagnostics
#endif

bool _Is_pod_comparator = n::__is_pod<int>::__value;
bool _Is_empty_check = n::__is_empty<int>::__value;

bool default_construct_int = n::is_trivially_constructible<int>::value;
bool copy_construct_int = n::is_trivially_constructible<int, const int&>::value;

// The built-ins should still work too:
bool _is_abstract_result = __is_abstract(int);
bool _is_arithmetic_result = __is_arithmetic(int);
bool _is_array_result = __is_array(int);
bool _is_assignable_result = __is_assignable(int, int);
bool _is_base_of_result = __is_base_of(int, int);
bool _is_class_result = __is_class(int);
bool _is_complete_type_result = __is_complete_type(int);
bool _is_compound_result = __is_compound(int);
bool _is_const_result = __is_const(int);
bool _is_constructible_result = __is_constructible(int);
bool _is_convertible_result = __is_convertible(int, int);
bool _is_convertible_to_result = __is_convertible_to(int, int);
bool _is_destructible_result = __is_destructible(int);
bool _is_empty_result = __is_empty(int);
bool _is_enum_result = __is_enum(int);
bool _is_floating_point_result = __is_floating_point(int);
bool _is_final_result = __is_final(int);
bool _is_function_result = __is_function(int);
bool _is_fundamental_result = __is_fundamental(int);
bool _is_integral_result = __is_integral(int);
bool _is_interface_class_result = __is_interface_class(int);
bool _is_literal_result = __is_literal(int);
bool _is_lvalue_expr_result = __is_lvalue_expr(0);
bool _is_lvalue_reference_result = __is_lvalue_reference(int);
bool _is_member_function_pointer_result = __is_member_function_pointer(int);
bool _is_member_object_pointer_result = __is_member_object_pointer(int);
bool _is_member_pointer_result = __is_member_pointer(int);
bool _is_nothrow_assignable_result = __is_nothrow_assignable(int, int);
bool _is_nothrow_constructible_result = __is_nothrow_constructible(int);
bool _is_nothrow_destructible_result = __is_nothrow_destructible(int);
bool _is_object_result = __is_object(int);
bool _is_pod_result = __is_pod(int);
bool _is_pointer_result = __is_pointer(int);
bool _is_polymorphic_result = __is_polymorphic(int);
bool _is_reference_result = __is_reference(int);
bool _is_rvalue_expr_result = __is_rvalue_expr(0);
bool _is_rvalue_reference_result = __is_rvalue_reference(int);
bool _is_same_result = __is_same(int, int);
bool _is_scalar_result = __is_scalar(int);
bool _is_sealed_result = __is_sealed(int);
bool _is_signed_result = __is_signed(int);
bool _is_standard_layout_result = __is_standard_layout(int);
bool _is_trivial_result = __is_trivial(int);
bool _is_trivially_assignable_result = __is_trivially_assignable(int, int);
bool _is_trivially_constructible_result = __is_trivially_constructible(int);
bool _is_trivially_copyable_result = __is_trivially_copyable(int);
bool _is_union_result = __is_union(int);
bool _is_unsigned_result = __is_unsigned(int);
bool _is_void_result = __is_void(int);
bool _is_volatile_result = __is_volatile(int);
