#!/usr/bin/env python

import os
from builtins import range
from functools import reduce

def get_libcxx_paths():
  utils_path = os.path.dirname(os.path.abspath(__file__))
  script_name = os.path.basename(__file__)
  assert os.path.exists(utils_path)
  src_root = os.path.dirname(utils_path)
  include_path = os.path.join(src_root, 'include')
  assert os.path.exists(include_path)
  docs_path = os.path.join(src_root, 'docs')
  assert os.path.exists(docs_path)
  macro_test_path = os.path.join(src_root, 'test', 'std', 'language.support',
                            'support.limits', 'support.limits.general')
  assert os.path.exists(macro_test_path)
  assert os.path.exists(os.path.join(macro_test_path, 'version.version.pass.cpp'))
  return script_name, src_root, include_path, docs_path, macro_test_path

script_name, source_root, include_path, docs_path, macro_test_path = get_libcxx_paths()

def has_header(h):
  h_path = os.path.join(include_path, h)
  return os.path.exists(h_path)

def add_version_header(tc):
  tc["headers"].append("version")
  return tc

feature_test_macros = [ add_version_header(x) for x in [
  {
    "name": "__cpp_lib_addressof_constexpr",
    "values": { "c++17": 201603 },
    "headers": ["memory"],
    "depends": "TEST_HAS_BUILTIN(__builtin_addressof) || TEST_GCC_VER >= 700",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_BUILTIN_ADDRESSOF)",
  }, {
    "name": "__cpp_lib_allocator_traits_is_always_equal",
    "values": { "c++17": 201411 },
    "headers": ["deque", "forward_list", "list", "map", "memory", "scoped_allocator", "set", "string", "unordered_map", "unordered_set", "vector"],
  }, {
    "name": "__cpp_lib_any",
    "values": { "c++17": 201606 },
    "headers": ["any"],
  }, {
    "name": "__cpp_lib_apply",
    "values": { "c++17": 201603 },
    "headers": ["tuple"],
  }, {
    "name": "__cpp_lib_array_constexpr",
    "values": { "c++17": 201603, "c++20": 201811 },
    "headers": ["array", "iterator"],
  }, {
    "name": "__cpp_lib_as_const",
    "values": { "c++17": 201510 },
    "headers": ["utility"],
  }, {
    "name": "__cpp_lib_assume_aligned",
    "values": { "c++20": 201811 },
    "headers": ["memory"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_atomic_flag_test",
    "values": { "c++20": 201907 },
    "headers": ["atomic"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
  }, {
    "name": "__cpp_lib_atomic_float",
    "values": { "c++20": 201711 },
    "headers": ["atomic"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_atomic_is_always_lock_free",
    "values": { "c++17": 201603 },
    "headers": ["atomic"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
  }, {
    "name": "__cpp_lib_atomic_lock_free_type_aliases",
    "values": { "c++20": 201907 },
    "headers": ["atomic"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
  }, {
    "name": "__cpp_lib_atomic_ref",
    "values": { "c++20": 201806 },
    "headers": ["atomic"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_atomic_shared_ptr",
    "values": { "c++20": 201711 },
    "headers": ["atomic"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_atomic_value_initialization",
    "values": { "c++20": 201911 },
    "headers": ["atomic", "memory"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_atomic_wait",
    "values": { "c++20": 201907 },
    "headers": ["atomic"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_atomic_wait)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_atomic_wait)",
  }, {
    "name": "__cpp_lib_barrier",
    "values": { "c++20": 201907 },
    "headers": ["barrier"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_barrier)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_barrier)",
  }, {
    "name": "__cpp_lib_bind_front",
    "values": { "c++20": 201907 },
    "headers": ["functional"],
  }, {
    "name": "__cpp_lib_bit_cast",
    "values": { "c++20": 201806 },
    "headers": ["bit"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_bitops",
    "values": { "c++20": 201907 },
    "headers": ["bit"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_bool_constant",
    "values": { "c++17": 201505 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_bounded_array_traits",
    "values": { "c++20": 201902 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_boyer_moore_searcher",
    "values": { "c++17": 201603 },
    "headers": ["functional"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_byte",
    "values": { "c++17": 201603 },
    "headers": ["cstddef"],
  }, {
    "name": "__cpp_lib_char8_t",
    "values": { "c++20": 201811 },
    "headers": ["atomic", "filesystem", "istream", "limits", "locale", "ostream", "string", "string_view"],
    "depends": "defined(__cpp_char8_t)",
    "internal_depends": "!defined(_LIBCPP_NO_HAS_CHAR8_T)",
  }, {
    "name": "__cpp_lib_chrono",
    "values": { "c++17": 201611 },
    "headers": ["chrono"],
  }, {
    "name": "__cpp_lib_chrono_udls",
    "values": { "c++14": 201304 },
    "headers": ["chrono"],
  }, {
    "name": "__cpp_lib_clamp",
    "values": { "c++17": 201603 },
    "headers": ["algorithm"],
  }, {
    "name": "__cpp_lib_complex_udls",
    "values": { "c++14": 201309 },
    "headers": ["complex"],
  }, {
    "name": "__cpp_lib_concepts",
    "values": { "c++20": 202002 },
    "headers": ["concepts"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_constexpr_algorithms",
    "values": { "c++20": 201806 },
    "headers": ["algorithm"],
  }, {
    "name": "__cpp_lib_constexpr_complex",
    "values": { "c++20": 201711 },
    "headers": ["complex"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_constexpr_dynamic_alloc",
    "values": { "c++20": 201907 },
    "headers": ["memory"],
  }, {
    "name": "__cpp_lib_constexpr_functional",
    "values": { "c++20": 201907 },
    "headers": ["functional"],
  }, {
    "name": "__cpp_lib_constexpr_iterator",
    "values": { "c++20": 201811 },
    "headers": ["iterator"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_constexpr_memory",
    "values": { "c++20": 201811 },
    "headers": ["memory"],
  }, {
    "name": "__cpp_lib_constexpr_numeric",
    "values": { "c++20": 201911 },
    "headers": ["numeric"],
  }, {
    "name": "__cpp_lib_constexpr_string",
    "values": { "c++20": 201907 },
    "headers": ["string"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_constexpr_string_view",
    "values": { "c++20": 201811 },
    "headers": ["string_view"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_constexpr_tuple",
    "values": { "c++20": 201811 },
    "headers": ["tuple"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_constexpr_utility",
    "values": { "c++20": 201811 },
    "headers": ["utility"],
  }, {
    "name": "__cpp_lib_constexpr_vector",
    "values": { "c++20": 201907 },
    "headers": ["vector"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_coroutine",
    "values": { "c++20": 201902 },
    "headers": ["coroutine"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_destroying_delete",
    "values": { "c++20": 201806 },
    "headers": ["new"],
    "depends": "TEST_STD_VER > 17 && defined(__cpp_impl_destroying_delete) && __cpp_impl_destroying_delete >= 201806L",
    "internal_depends": "_LIBCPP_STD_VER > 17 && defined(__cpp_impl_destroying_delete) && __cpp_impl_destroying_delete >= 201806L",
  }, {
    "name": "__cpp_lib_enable_shared_from_this",
    "values": { "c++17": 201603 },
    "headers": ["memory"],
  }, {
    "name": "__cpp_lib_endian",
    "values": { "c++20": 201907 },
    "headers": ["bit"],
  }, {
    "name": "__cpp_lib_erase_if",
    "values": { "c++20": 202002 },
    "headers": ["deque", "forward_list", "list", "map", "set", "string", "unordered_map", "unordered_set", "vector"],
  }, {
    "name": "__cpp_lib_exchange_function",
    "values": { "c++14": 201304 },
    "headers": ["utility"],
  }, {
    "name": "__cpp_lib_execution",
    "values": { "c++17": 201603, "c++20": 201902 },
    "headers": ["execution"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_filesystem",
    "values": { "c++17": 201703 },
    "headers": ["filesystem"],
    "depends": "!defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_filesystem)",
    "internal_depends": "!defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_filesystem)"
  }, {
    "name": "__cpp_lib_format",
    "values": { "c++20": 201907 },
    "headers": ["format"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_gcd_lcm",
    "values": { "c++17": 201606 },
    "headers": ["numeric"],
  }, {
    "name": "__cpp_lib_generic_associative_lookup",
    "values": { "c++14": 201304 },
    "headers": ["map", "set"],
  }, {
    "name": "__cpp_lib_generic_unordered_lookup",
    "values": { "c++20": 201811 },
    "headers": ["unordered_map", "unordered_set"],
  }, {
    "name": "__cpp_lib_hardware_interference_size",
    "values": { "c++17": 201703 },
    "headers": ["new"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_has_unique_object_representations",
    "values": { "c++17": 201606 },
    "headers": ["type_traits"],
    "depends": "TEST_HAS_BUILTIN_IDENTIFIER(__has_unique_object_representations) || TEST_GCC_VER >= 700",
    "internal_depends": "defined(_LIBCPP_HAS_UNIQUE_OBJECT_REPRESENTATIONS)",
  }, {
    "name": "__cpp_lib_hypot",
    "values": { "c++17": 201603 },
    "headers": ["cmath"],
  }, {
    "name": "__cpp_lib_incomplete_container_elements",
    "values": { "c++17": 201505 },
    "headers": ["forward_list", "list", "vector"],
  }, {
    "name": "__cpp_lib_int_pow2",
    "values": { "c++20": 202002 },
    "headers": ["bit"],
  }, {
    "name": "__cpp_lib_integer_comparison_functions",
    "values": { "c++20": 202002 },
    "headers": ["utility"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_integer_sequence",
    "values": { "c++14": 201304 },
    "headers": ["utility"],
  }, {
    "name": "__cpp_lib_integral_constant_callable",
    "values": { "c++14": 201304 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_interpolate",
    "values": { "c++20": 201902 },
    "headers": ["cmath", "numeric"],
  }, {
    "name": "__cpp_lib_invoke",
    "values": { "c++17": 201411 },
    "headers": ["functional"],
  }, {
    "name": "__cpp_lib_is_aggregate",
    "values": { "c++17": 201703 },
    "headers": ["type_traits"],
    "depends": "TEST_HAS_BUILTIN_IDENTIFIER(__is_aggregate) || TEST_GCC_VER_NEW >= 7001",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_IS_AGGREGATE)",
  }, {
    "name": "__cpp_lib_is_constant_evaluated",
    "values": { "c++20": 201811 },
    "headers": ["type_traits"],
    "depends": "TEST_HAS_BUILTIN(__builtin_is_constant_evaluated) || TEST_GCC_VER >= 900",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_BUILTIN_IS_CONSTANT_EVALUATED)",
  }, {
    "name": "__cpp_lib_is_final",
    "values": { "c++14": 201402 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_is_invocable",
    "values": { "c++17": 201703 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_is_layout_compatible",
    "values": { "c++20": 201907 },
    "headers": ["type_traits"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_is_nothrow_convertible",
    "values": { "c++20": 201806 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_is_null_pointer",
    "values": { "c++14": 201309 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_is_pointer_interconvertible",
    "values": { "c++20": 201907 },
    "headers": ["type_traits"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_is_scoped_enum",
    "values": { "c++2b": 202011 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_is_swappable",
    "values": { "c++17": 201603 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_jthread",
    "values": { "c++20": 201911 },
    "headers": ["stop_token", "thread"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_latch",
    "values": { "c++20": 201907 },
    "headers": ["latch"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_latch)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_latch)",
  }, {
    "name": "__cpp_lib_launder",
    "values": { "c++17": 201606 },
    "headers": ["new"],
  }, {
    "name": "__cpp_lib_list_remove_return_type",
    "values": { "c++20": 201806 },
    "headers": ["forward_list", "list"],
  }, {
    "name": "__cpp_lib_logical_traits",
    "values": { "c++17": 201510 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_make_from_tuple",
    "values": { "c++17": 201606 },
    "headers": ["tuple"],
  }, {
    "name": "__cpp_lib_make_reverse_iterator",
    "values": { "c++14": 201402 },
    "headers": ["iterator"],
  }, {
    "name": "__cpp_lib_make_unique",
    "values": { "c++14": 201304 },
    "headers": ["memory"],
  }, {
    "name": "__cpp_lib_map_try_emplace",
    "values": { "c++17": 201411 },
    "headers": ["map"],
  }, {
    "name": "__cpp_lib_math_constants",
    "values": { "c++20": 201907 },
    "headers": ["numbers"],
    "depends": "defined(__cpp_concepts) && __cpp_concepts >= 201907L",
    "internal_depends": "defined(__cpp_concepts) && __cpp_concepts >= 201907L",
  }, {
    "name": "__cpp_lib_math_special_functions",
    "values": { "c++17": 201603 },
    "headers": ["cmath"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_memory_resource",
    "values": { "c++17": 201603 },
    "headers": ["memory_resource"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_node_extract",
    "values": { "c++17": 201606 },
    "headers": ["map", "set", "unordered_map", "unordered_set"],
  }, {
    "name": "__cpp_lib_nonmember_container_access",
    "values": { "c++17": 201411 },
    "headers": ["array", "deque", "forward_list", "iterator", "list", "map", "regex", "set", "string", "unordered_map", "unordered_set", "vector"],
  }, {
    "name": "__cpp_lib_not_fn",
    "values": { "c++17": 201603 },
    "headers": ["functional"],
  }, {
    "name": "__cpp_lib_null_iterators",
    "values": { "c++14": 201304 },
    "headers": ["iterator"],
  }, {
    "name": "__cpp_lib_optional",
    "values": { "c++17": 201606 },
    "headers": ["optional"],
  }, {
    "name": "__cpp_lib_parallel_algorithm",
    "values": { "c++17": 201603 },
    "headers": ["algorithm", "numeric"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_polymorphic_allocator",
    "values": { "c++20": 201902 },
    "headers": ["memory"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_quoted_string_io",
    "values": { "c++14": 201304 },
    "headers": ["iomanip"],
  }, {
    "name": "__cpp_lib_ranges",
    "values": { "c++20": 201811 },
    "headers": ["algorithm", "functional", "iterator", "memory", "ranges"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_raw_memory_algorithms",
    "values": { "c++17": 201606 },
    "headers": ["memory"],
  }, {
    "name": "__cpp_lib_remove_cvref",
    "values": { "c++20": 201711 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_result_of_sfinae",
    "values": { "c++14": 201210 },
    "headers": ["functional", "type_traits"],
  }, {
    "name": "__cpp_lib_robust_nonmodifying_seq_ops",
    "values": { "c++14": 201304 },
    "headers": ["algorithm"],
  }, {
    "name": "__cpp_lib_sample",
    "values": { "c++17": 201603 },
    "headers": ["algorithm"],
  }, {
    "name": "__cpp_lib_scoped_lock",
    "values": { "c++17": 201703 },
    "headers": ["mutex"],
  }, {
    "name": "__cpp_lib_semaphore",
    "values": { "c++20": 201907 },
    "headers": ["semaphore"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_semaphore)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_semaphore)",
  }, {
    "name": "__cpp_lib_shared_mutex",
    "values": { "c++17": 201505 },
    "headers": ["shared_mutex"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_shared_mutex)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_shared_mutex)",
  }, {
    "name": "__cpp_lib_shared_ptr_arrays",
    "values": { "c++17": 201611 },
    "headers": ["memory"],
  }, {
    "name": "__cpp_lib_shared_ptr_weak_type",
    "values": { "c++17": 201606 },
    "headers": ["memory"],
  }, {
    "name": "__cpp_lib_shared_timed_mutex",
    "values": { "c++14": 201402 },
    "headers": ["shared_mutex"],
    "depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_shared_timed_mutex)",
    "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS) && !defined(_LIBCPP_AVAILABILITY_DISABLE_FTM___cpp_lib_shared_timed_mutex)",
  }, {
    "name": "__cpp_lib_shift",
    "values": { "c++20": 201806 },
    "headers": ["algorithm"],
  }, {
    "name": "__cpp_lib_smart_ptr_for_overwrite",
    "values": { "c++20": 202002 },
    "headers": ["memory"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_source_location",
    "values": { "c++20": 201907 },
    "headers": ["source_location"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_span",
    "values": { "c++20": 202002 },
    "headers": ["span"],
  }, {
    "name": "__cpp_lib_ssize",
    "values": { "c++20": 201902 },
    "headers": ["iterator"],
  }, {
    "name": "__cpp_lib_stacktrace",
    "values": { "c++2b": 202011 },
    "headers": ["stacktrace"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_starts_ends_with",
    "values": { "c++20": 201711 },
    "headers": ["string", "string_view"],
  }, {
    "name": "__cpp_lib_stdatomic_h",
    "values": { "c++2b": 202011 },
    "headers": ["stdatomic.h"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_string_contains",
    "values": { "c++2b": 202011 },
    "headers": ["string", "string_view"],
  }, {
    "name": "__cpp_lib_string_udls",
    "values": { "c++14": 201304 },
    "headers": ["string"],
  }, {
    "name": "__cpp_lib_string_view",
    "values": { "c++17": 201606, "c++20": 201803 },
    "headers": ["string", "string_view"],
  }, {
    "name": "__cpp_lib_syncbuf",
    "values": { "c++20": 201803 },
    "headers": ["syncstream"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_three_way_comparison",
    "values": { "c++20": 201907 },
    "headers": ["compare"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_to_address",
    "values": { "c++20": 201711 },
    "headers": ["memory"],
  }, {
    "name": "__cpp_lib_to_array",
    "values": { "c++20": 201907 },
    "headers": ["array"],
  }, {
    "name": "__cpp_lib_to_chars",
    "values": { "c++17": 201611 },
    "headers": ["utility"],
    "unimplemented": True,
  }, {
    "name": "__cpp_lib_to_underlying",
    "values": { "c++2b": 202102 },
    "headers": ["utility"],
  }, {
    "name": "__cpp_lib_transformation_trait_aliases",
    "values": { "c++14": 201304 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_transparent_operators",
    "values": { "c++14": 201210, "c++17": 201510 },
    "headers": ["functional", "memory"],
  }, {
    "name": "__cpp_lib_tuple_element_t",
    "values": { "c++14": 201402 },
    "headers": ["tuple"],
  }, {
    "name": "__cpp_lib_tuples_by_type",
    "values": { "c++14": 201304 },
    "headers": ["tuple", "utility"],
  }, {
    "name": "__cpp_lib_type_trait_variable_templates",
    "values": { "c++17": 201510 },
    "headers": ["type_traits"],
  }, {
    "name": "__cpp_lib_uncaught_exceptions",
    "values": { "c++17": 201411 },
    "headers": ["exception"],
  }, {
    "name": "__cpp_lib_unordered_map_try_emplace",
    "values": { "c++17": 201411 },
    "headers": ["unordered_map"],
  }, {
    "name": "__cpp_lib_unwrap_ref",
    "values": { "c++20": 201811 },
    "headers": ["functional"],
  }, {
    "name": "__cpp_lib_variant",
    "values": { "c++17": 201606 },
    "headers": ["variant"],
  }, {
    "name": "__cpp_lib_void_t",
    "values": { "c++17": 201411 },
    "headers": ["type_traits"],
  }
]]

assert feature_test_macros == sorted(feature_test_macros, key=lambda tc: tc["name"])
assert all(tc["headers"] == sorted(tc["headers"]) for tc in feature_test_macros)

# Map from each header to the Lit annotations that should be used for
# tests that include that header.
#
# For example, when threads are not supported, any feature-test-macro test
# that includes <thread> should be marked as UNSUPPORTED, because including
# <thread> is a hard error in that case.
lit_markup = {
  "atomic": ["UNSUPPORTED: libcpp-has-no-threads"],
  "barrier": ["UNSUPPORTED: libcpp-has-no-threads"],
  "filesystem": ["UNSUPPORTED: libcpp-has-no-filesystem-library"],
  "iomanip": ["UNSUPPORTED: libcpp-has-no-localization"],
  "istream": ["UNSUPPORTED: libcpp-has-no-localization"],
  "latch": ["UNSUPPORTED: libcpp-has-no-threads"],
  "locale": ["UNSUPPORTED: libcpp-has-no-localization"],
  "ostream": ["UNSUPPORTED: libcpp-has-no-localization"],
  "regex": ["UNSUPPORTED: libcpp-has-no-localization"],
  "semaphore": ["UNSUPPORTED: libcpp-has-no-threads"],
  "shared_mutex": ["UNSUPPORTED: libcpp-has-no-threads"],
  "thread": ["UNSUPPORTED: libcpp-has-no-threads"],
}

def get_std_dialects():
  std_dialects = ['c++14', 'c++17', 'c++20', 'c++2b']
  return list(std_dialects)

def get_first_std(d):
    for s in get_std_dialects():
        if s in d.keys():
            return s
    return None

def get_last_std(d):
  rev_dialects = get_std_dialects()
  rev_dialects.reverse()
  for s in rev_dialects:
    if s in d.keys():
      return s
  return None

def get_std_before(d, std):
  std_dialects = get_std_dialects()
  candidates = std_dialects[0:std_dialects.index(std)]
  candidates.reverse()
  for cand in candidates:
    if cand in d.keys():
      return cand
  return None

def get_value_before(d, std):
  new_std = get_std_before(d, std)
  if new_std is None:
    return None
  return d[new_std]

def get_for_std(d, std):
  # This catches the C++11 case for which there should be no defined feature
  # test macros.
  std_dialects = get_std_dialects()
  if std not in std_dialects:
    return None
  # Find the value for the newest C++ dialect between C++14 and std
  std_list = list(std_dialects[0:std_dialects.index(std)+1])
  std_list.reverse()
  for s in std_list:
    if s in d.keys():
      return d[s]
  return None

def get_std_number(std):
    return std.replace('c++', '')

"""
  Functions to produce the <version> header
"""

def produce_macros_definition_for_std(std):
  result = ""
  indent = 56
  for tc in feature_test_macros:
    if std not in tc["values"]:
      continue
    inner_indent = 1
    if 'depends' in tc.keys():
      assert 'internal_depends' in tc.keys()
      result += "# if %s\n" % tc["internal_depends"]
      inner_indent += 2
    if get_value_before(tc["values"], std) is not None:
      assert 'depends' not in tc.keys()
      result += "# undef  %s\n" % tc["name"]
    line = "#%sdefine %s" % ((" " * inner_indent), tc["name"])
    line += " " * (indent - len(line))
    line += "%sL" % tc["values"][std]
    if 'unimplemented' in tc.keys():
      line = "// " + line
    result += line
    result += "\n"
    if 'depends' in tc.keys():
      result += "# endif\n"
  return result.strip()

def produce_macros_definitions():
  macro_definition_template = """#if _LIBCPP_STD_VER > {previous_std_number}
{macro_definition}
#endif"""

  macros_definitions = []
  previous_std_number = '11'
  for std in get_std_dialects():
    macros_definitions.append(
      macro_definition_template.format(previous_std_number=previous_std_number,
                                       macro_definition=produce_macros_definition_for_std(std)))
    previous_std_number = get_std_number(std)

  return '\n\n'.join(macros_definitions)

def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]

def produce_version_synopsis():
  indent = 56
  header_indent = 56 + len("20XXYYL ")
  result = ""
  def indent_to(s, val):
    if len(s) >= val:
      return s
    s += " " * (val - len(s))
    return s
  line = indent_to("Macro name", indent) + "Value"
  line = indent_to(line, header_indent) + "Headers"
  result += line + "\n"
  for tc in feature_test_macros:
    prev_defined_std = get_last_std(tc["values"])
    line = "{name: <{indent}}{value}L ".format(name=tc['name'], indent=indent,
                                               value=tc["values"][prev_defined_std])
    headers = list(tc["headers"])
    headers.remove("version")
    for chunk in chunks(headers, 3):
      line = indent_to(line, header_indent)
      chunk = ['<%s>' % header for header in chunk]
      line += ' '.join(chunk)
      result += line
      result += "\n"
      line = ""
    while True:
      prev_defined_std = get_std_before(tc["values"], prev_defined_std)
      if prev_defined_std is None:
        break
      result += "%s%sL // %s\n" % (indent_to("", indent), tc["values"][prev_defined_std],
                                prev_defined_std.replace("c++", "C++"))
  return result


def produce_version_header():
  template="""// -*- C++ -*-
//===--------------------------- version ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_VERSIONH
#define _LIBCPP_VERSIONH

/*
  version synopsis

{synopsis}

*/

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

// clang-format off

{cxx_macros}

// clang-format on

#endif // _LIBCPP_VERSIONH
"""

  version_str = template.format(
      synopsis=produce_version_synopsis().strip(),
      cxx_macros=produce_macros_definitions())
  version_header_path = os.path.join(include_path, 'version')
  with open(version_header_path, 'w', newline='\n') as f:
    f.write(version_str)


"""
    Functions to produce test files
"""

test_types = {
  "undefined": """
# ifdef {name}
#   error "{name} should not be defined before {std_first}"
# endif
""",

  "depends": """
# if {depends}
#   ifndef {name}
#     error "{name} should be defined in {std}"
#   endif
#   if {name} != {value}
#     error "{name} should have the value {value} in {std}"
#   endif
# else
#   ifdef {name}
#     error "{name} should not be defined when {depends} is not defined!"
#   endif
# endif
""",

  "unimplemented": """
# if !defined(_LIBCPP_VERSION)
#   ifndef {name}
#     error "{name} should be defined in {std}"
#   endif
#   if {name} != {value}
#     error "{name} should have the value {value} in {std}"
#   endif
# else // _LIBCPP_VERSION
#   ifdef {name}
#     error "{name} should not be defined because it is unimplemented in libc++!"
#   endif
# endif
""",

  "defined": """
# ifndef {name}
#   error "{name} should be defined in {std}"
# endif
# if {name} != {value}
#   error "{name} should have the value {value} in {std}"
# endif
"""
}

def generate_std_test(test_list, std):
  result = ""
  for tc in test_list:
    val = get_for_std(tc["values"], std)
    if val is not None:
      val = "%sL" % val
    if val is None:
      result += test_types["undefined"].format(name=tc["name"], std_first=get_first_std(tc["values"]))
    elif 'unimplemented' in tc.keys():
      result += test_types["unimplemented"].format(name=tc["name"], value=val, std=std)
    elif "depends" in tc.keys():
      result += test_types["depends"].format(name=tc["name"], value=val, std=std, depends=tc["depends"])
    else:
      result +=  test_types["defined"].format(name=tc["name"], value=val, std=std)
  return result.strip()

def generate_std_tests(test_list):
  std_tests_template = """#if TEST_STD_VER < {first_std_number}

{pre_std_test}

{other_std_tests}

#elif TEST_STD_VER > {penultimate_std_number}

{last_std_test}

#endif // TEST_STD_VER > {penultimate_std_number}"""

  std_dialects = get_std_dialects()
  assert not get_std_number(std_dialects[-1]).isnumeric()

  other_std_tests = []
  for std in std_dialects[:-1]:
    other_std_tests.append('#elif TEST_STD_VER == ' + get_std_number(std))
    other_std_tests.append(generate_std_test(test_list, std))

  std_tests = std_tests_template.format(first_std_number=get_std_number(std_dialects[0]),
                                        pre_std_test=generate_std_test(test_list, 'c++11'),
                                        other_std_tests='\n\n'.join(other_std_tests),
                                        penultimate_std_number=get_std_number(std_dialects[-2]),
                                        last_std_test=generate_std_test(test_list, std_dialects[-1]))

  return std_tests

def generate_synopsis(test_list):
    max_name_len = max([len(tc["name"]) for tc in test_list])
    indent = max_name_len + 8
    def mk_line(prefix, suffix):
        return "{prefix: <{max_len}}{suffix}\n".format(prefix=prefix, suffix=suffix,
        max_len=indent)
    result = ""
    result += mk_line("/*  Constant", "Value")
    for tc in test_list:
        prefix = "    %s" % tc["name"]
        for std in [s for s in get_std_dialects() if s in tc["values"].keys()]:
            result += mk_line(prefix, "%sL [%s]" % (tc["values"][std], std.replace("c++", "C++")))
            prefix = ""
    result += "*/"
    return result

def produce_tests():
  headers = set([h for tc in feature_test_macros for h in tc["headers"]])
  for h in headers:
    test_list = [tc for tc in feature_test_macros if h in tc["headers"]]
    if not has_header(h):
      for tc in test_list:
        assert 'unimplemented' in tc.keys()
      continue
    markup = '\n'.join('// ' + tag for tag in lit_markup.get(h, []))
    test_body = \
"""//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// WARNING: This test was generated by {script_name}
// and should not be edited manually.
//
// clang-format off
{markup}
// <{header}>

// Test the feature test macros defined by <{header}>

{synopsis}

#include <{header}>
#include "test_macros.h"

{cxx_tests}

int main(int, char**) {{ return 0; }}
""".format(script_name=script_name,
           header=h,
           markup=('\n{}\n'.format(markup) if markup else ''),
           synopsis=generate_synopsis(test_list),
           cxx_tests=generate_std_tests(test_list))
    test_name = "{header}.version.pass.cpp".format(header=h)
    out_path = os.path.join(macro_test_path, test_name)
    with open(out_path, 'w', newline='\n') as f:
      f.write(test_body)

"""
    Produce documentation for the feature test macros
"""

def make_widths(grid):
  widths = []
  for i in range(0, len(grid[0])):
    cell_width = 2 + max(reduce(lambda x,y: x+y, [[len(row[i])] for row in grid], []))
    widths += [cell_width]
  return widths

def create_table(grid, indent):
  indent_str = ' '*indent
  col_widths = make_widths(grid)
  result = [indent_str + add_divider(col_widths, 2)]
  header_flag = 2
  for row_i in range(0, len(grid)):
    row = grid[row_i]
    line = indent_str + ' '.join([pad_cell(row[i], col_widths[i]) for i in range(0, len(row))])
    result.append(line.rstrip())
    is_cxx_header = row[0].startswith('**')
    if row_i == len(grid) - 1:
      header_flag = 2
    separator = indent_str + add_divider(col_widths, 1 if is_cxx_header else header_flag)
    result.append(separator.rstrip())
    header_flag = 0
  return '\n'.join(result)

def add_divider(widths, header_flag):
  if header_flag == 2:
    return ' '.join(['='*w for w in widths])
  if header_flag == 1:
    return '-'.join(['-'*w for w in widths])
  else:
    return ' '.join(['-'*w for w in widths])

def pad_cell(s, length, left_align=True):
  padding = ((length - len(s)) * ' ')
  return s + padding


def get_status_table():
  table = [["Macro Name", "Value"]]
  for std in get_std_dialects():
    table += [["**" + std.replace("c++", "C++ ") + "**", ""]]
    for tc in feature_test_macros:
      if std not in tc["values"].keys():
        continue
      value = "``%sL``" % tc["values"][std]
      if 'unimplemented' in tc.keys():
        value = '*unimplemented*'
      table += [["``%s``" % tc["name"], value]]
  return table

def produce_docs():
  doc_str = """.. _FeatureTestMacroTable:

==========================
Feature Test Macro Support
==========================

.. contents::
   :local:

Overview
========

This file documents the feature test macros currently supported by libc++.

.. _feature-status:

Status
======

.. table:: Current Status
     :name: feature-status-table
     :widths: auto

{status_tables}

""".format(status_tables=create_table(get_status_table(), 4))

  table_doc_path = os.path.join(docs_path, 'FeatureTestMacroTable.rst')
  with open(table_doc_path, 'w', newline='\n') as f:
    f.write(doc_str)

def main():
  produce_version_header()
  produce_tests()
  produce_docs()


if __name__ == '__main__':
  main()
