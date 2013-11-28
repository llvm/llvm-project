// RUN: %clang_cc1 -std=c++98 -verify %s
// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++1y -verify %s

// expected-no-diagnostics

#if __cplusplus < 201103L
#define check(macro, cxx98, cxx11, cxx1y) cxx98 == 0 ? defined(__cpp_##macro) : __cpp_##macro != cxx98
#elif __cplusplus < 201304L
#define check(macro, cxx98, cxx11, cxx1y) cxx11 == 0 ? defined(__cpp_##macro) : __cpp_##macro != cxx11
#else
#define check(macro, cxx98, cxx11, cxx1y) cxx1y == 0 ? defined(__cpp_##macro) : __cpp_##macro != cxx1y
#endif

#if check(binary_literals, 0, 0, 201304)
#error "wrong value for __cpp_binary_literals"
#endif

#if check(init_captures, 0, 0, 201304)
#error "wrong value for __cpp_init_captures"
#endif

#if check(generic_lambdas, 0, 0, 201304)
#error "wrong value for __cpp_generic_lambdas"
#endif

#if check(constexpr, 0, 200704, 201304)
#error "wrong value for __cpp_constexpr"
#endif

#if check(decltype_auto, 0, 0, 201304)
#error "wrong value for __cpp_decltype_auto"
#endif

#if check(return_type_deduction, 0, 0, 201304)
#error "wrong value for __cpp_return_type_deduction"
#endif

#if check(runtime_arrays, 0, 0, 0)
#error "wrong value for __cpp_runtime_arrays"
#endif

#if check(aggregate_nsdmi, 0, 0, 201304)
#error "wrong value for __cpp_aggregate_nsdmi"
#endif

#if check(variable_templates, 0, 0, 201304)
#error "wrong value for __cpp_variable_templates"
#endif

#if check(unicode_characters, 0, 200704, 200704)
#error "wrong value for __cpp_unicode_characters"
#endif

#if check(raw_strings, 0, 200710, 200710)
#error "wrong value for __cpp_raw_strings"
#endif

#if check(unicode_literals, 0, 200710, 200710)
#error "wrong value for __cpp_unicode_literals"
#endif

#if check(user_defined_literals, 0, 200809, 200809)
#error "wrong value for __cpp_user_defined_literals"
#endif

#if check(lambdas, 0, 200907, 200907)
#error "wrong value for __cpp_lambdas"
#endif

#if check(static_assert, 0, 200410, 200410)
#error "wrong value for __cpp_static_assert"
#endif

#if check(decltype, 0, 200707, 200707)
#error "wrong value for __cpp_decltype"
#endif

#if check(attributes, 0, 200809, 200809)
#error "wrong value for __cpp_attributes"
#endif

#if check(rvalue_references, 0, 200610, 200610)
#error "wrong value for __cpp_rvalue_references"
#endif

#if check(variadic_templates, 0, 200704, 200704)
#error "wrong value for __cpp_variadic_templates"
#endif
