// RUN: %clang_cc1 -std=c++98 -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++1y -fcxx-exceptions -fsized-deallocation -verify %s
// RUN: %clang_cc1 -std=c++14 -fcxx-exceptions -fsized-deallocation -verify %s
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fsized-deallocation -verify %s
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fsized-deallocation -fconcepts-ts -DCONCEPTS_TS=1 -verify %s
// RUN: %clang_cc1 -fno-rtti -fno-threadsafe-statics -verify %s -DNO_EXCEPTIONS -DNO_RTTI -DNO_THREADSAFE_STATICS
// RUN: %clang_cc1 -fcoroutines-ts -DNO_EXCEPTIONS -DCOROUTINES -verify %s

// expected-no-diagnostics

// FIXME using `defined` in a macro has undefined behavior.
#if __cplusplus < 201103L
#define check(macro, cxx98, cxx11, cxx14, cxx1z) cxx98 == 0 ? defined(__cpp_##macro) : __cpp_##macro != cxx98
#elif __cplusplus < 201402L
#define check(macro, cxx98, cxx11, cxx14, cxx1z) cxx11 == 0 ? defined(__cpp_##macro) : __cpp_##macro != cxx11
#elif __cplusplus < 201406L
#define check(macro, cxx98, cxx11, cxx14, cxx1z) cxx14 == 0 ? defined(__cpp_##macro) : __cpp_##macro != cxx14
#else
#define check(macro, cxx98, cxx11, cxx14, cxx1z) cxx1z == 0 ? defined(__cpp_##macro) : __cpp_##macro != cxx1z
#endif

// --- C++17 features ---

#if check(hex_float, 0, 0, 0, 201603)
#error "wrong value for __cpp_hex_float"
#endif

#if check(inline_variables, 0, 0, 0, 201606)
#error "wrong value for __cpp_inline_variables"
#endif

#if check(aligned_new, 0, 0, 0, 201606)
#error "wrong value for __cpp_aligned_new"
#endif

#if check(noexcept_function_type, 0, 0, 0, 201510)
#error "wrong value for __cpp_noexcept_function_type"
#endif

#if check(fold_expressions, 0, 0, 0, 201603)
#error "wrong value for __cpp_fold_expressions"
#endif

#if check(capture_star_this, 0, 0, 0, 201603)
#error "wrong value for __cpp_capture_star_this"
#endif

// constexpr checked below

#if check(if_constexpr, 0, 0, 0, 201606)
#error "wrong value for __cpp_if_constexpr"
#endif

// range_based_for checked below

// static_assert checked below

#if check(deduction_guides, 0, 0, 0, 201611)
#error "wrong value for __cpp_deduction_guides"
#endif

#if check(template_auto, 0, 0, 0, 201606)
#error "wrong value for __cpp_template_auto"
#endif

#if check(namespace_attributes, 0, 0, 0, 201411)
// FIXME: allowed without warning in C++14 and C++11
#error "wrong value for __cpp_namespace_attributes"
#endif

#if check(enumerator_attributes, 0, 0, 0, 201411)
// FIXME: allowed without warning in C++14 and C++11
#error "wrong value for __cpp_enumerator_attributes"
#endif

#if check(nested_namespace_definitions, 0, 0, 0, 201411)
#error "wrong value for __cpp_nested_namespace_definitions"
#endif

// inheriting_constructors checked below

#if check(variadic_using, 0, 0, 0, 201611)
#error "wrong value for __cpp_variadic_using"
#endif

#if check(aggregate_bases, 0, 0, 0, 201603)
#error "wrong value for __cpp_aggregate_bases"
#endif

#if check(structured_bindings, 0, 0, 0, 201606)
#error "wrong value for __cpp_structured_bindings"
#endif

#if check(nontype_template_args, 0, 0, 0, 201411)
#error "wrong value for __cpp_nontype_template_args"
#endif

#if check(template_template_args, 0, 0, 0, 0) // FIXME: should be 201611 when feature is enabled
#error "wrong value for __cpp_template_template_args"
#endif

// --- C++14 features ---

#if check(binary_literals, 0, 0, 201304, 201304)
#error "wrong value for __cpp_binary_literals"
#endif

// (Removed from SD-6.)
#if check(digit_separators, 0, 0, 201309, 201309)
#error "wrong value for __cpp_digit_separators"
#endif

#if check(init_captures, 0, 0, 201304, 201304)
#error "wrong value for __cpp_init_captures"
#endif

#if check(generic_lambdas, 0, 0, 201304, 201304)
#error "wrong value for __cpp_generic_lambdas"
#endif

#if check(sized_deallocation, 0, 0, 201309, 201309)
#error "wrong value for __cpp_sized_deallocation"
#endif

// constexpr checked below

#if check(decltype_auto, 0, 0, 201304, 201304)
#error "wrong value for __cpp_decltype_auto"
#endif

#if check(return_type_deduction, 0, 0, 201304, 201304)
#error "wrong value for __cpp_return_type_deduction"
#endif

#if check(runtime_arrays, 0, 0, 0, 0)
#error "wrong value for __cpp_runtime_arrays"
#endif

#if check(aggregate_nsdmi, 0, 0, 201304, 201304)
#error "wrong value for __cpp_aggregate_nsdmi"
#endif

#if check(variable_templates, 0, 0, 201304, 201304)
#error "wrong value for __cpp_variable_templates"
#endif

// --- C++11 features ---

#if check(unicode_characters, 0, 200704, 200704, 200704)
#error "wrong value for __cpp_unicode_characters"
#endif

#if check(raw_strings, 0, 200710, 200710, 200710)
#error "wrong value for __cpp_raw_strings"
#endif

#if check(unicode_literals, 0, 200710, 200710, 200710)
#error "wrong value for __cpp_unicode_literals"
#endif

#if check(user_defined_literals, 0, 200809, 200809, 200809)
#error "wrong value for __cpp_user_defined_literals"
#endif

#if defined(NO_THREADSAFE_STATICS) ? check(threadsafe_static_init, 0, 0, 0, 0) : check(threadsafe_static_init, 200806, 200806, 200806, 200806)
#error "wrong value for __cpp_threadsafe_static_init"
#endif

#if check(lambdas, 0, 200907, 200907, 200907)
#error "wrong value for __cpp_lambdas"
#endif

#if check(constexpr, 0, 200704, 201304, 201603)
#error "wrong value for __cpp_constexpr"
#endif

#if check(range_based_for, 0, 200907, 200907, 201603)
#error "wrong value for __cpp_range_based_for"
#endif

#if check(static_assert, 0, 200410, 200410, 201411)
#error "wrong value for __cpp_static_assert"
#endif

#if check(decltype, 0, 200707, 200707, 200707)
#error "wrong value for __cpp_decltype"
#endif

#if check(attributes, 0, 200809, 200809, 200809)
#error "wrong value for __cpp_attributes"
#endif

#if check(rvalue_references, 0, 200610, 200610, 200610)
#error "wrong value for __cpp_rvalue_references"
#endif

#if check(variadic_templates, 0, 200704, 200704, 200704)
#error "wrong value for __cpp_variadic_templates"
#endif

#if check(initializer_lists, 0, 200806, 200806, 200806)
#error "wrong value for __cpp_initializer_lists"
#endif

#if check(delegating_constructors, 0, 200604, 200604, 200604)
#error "wrong value for __cpp_delegating_constructors"
#endif

#if check(nsdmi, 0, 200809, 200809, 200809)
#error "wrong value for __cpp_nsdmi"
#endif

#if check(inheriting_constructors, 0, 201511, 201511, 201511)
#error "wrong value for __cpp_inheriting_constructors"
#endif

#if check(ref_qualifiers, 0, 200710, 200710, 200710)
#error "wrong value for __cpp_ref_qualifiers"
#endif

#if check(alias_templates, 0, 200704, 200704, 200704)
#error "wrong value for __cpp_alias_templates"
#endif

// --- C++98 features ---

#if defined(NO_RTTI) ? check(rtti, 0, 0, 0, 0) : check(rtti, 199711, 199711, 199711, 199711)
#error "wrong value for __cpp_rtti"
#endif

#if defined(NO_EXCEPTIONS) ? check(exceptions, 0, 0, 0, 0) : check(exceptions, 199711, 199711, 199711, 199711)
#error "wrong value for __cpp_exceptions"
#endif

// --- TS features --

#if check(experimental_concepts, 0, 0, CONCEPTS_TS, CONCEPTS_TS)
#error "wrong value for __cpp_experimental_concepts"
#endif

#if defined(COROUTINES) ? check(coroutines, 201703L, 201703L, 201703L, 201703L) : check(coroutines, 0, 0, 0, 0)
#error "wrong value for __cpp_coroutines"
#endif
