#!/usr/bin/env python

import os
import tempfile
from builtins import int, range
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

feature_test_macros = sorted([ add_version_header(x) for x in [
  # C++14 macros
  {"name": "__cpp_lib_integer_sequence",
   "values": {
      "c++14": int(201304)
    },
    "headers": ["utility"],
  },
  {"name": "__cpp_lib_exchange_function",
   "values": {
     "c++14": int(201304)
   },
   "headers": ["utility"],
  },
  {"name": "__cpp_lib_tuples_by_type",
   "values": {
     "c++14": int(201304)
   },
   "headers": ["utility", "tuple"],
  },
  {"name": "__cpp_lib_tuple_element_t",
   "values": {
     "c++14": int(201402)
   },
   "headers": ["tuple"],
  },
  {"name": "__cpp_lib_make_unique",
   "values": {
     "c++14": int(201304)
   },
   "headers": ["memory"],
  },
  {"name": "__cpp_lib_transparent_operators",
   "values": {
     "c++14": int(201210),
     "c++17": int(201510),
   },
   "headers": ["functional"],
  },
  {"name": "__cpp_lib_integral_constant_callable",
   "values": {
     "c++14": int(201304)
   },
   "headers": ["type_traits"],
  },
  {"name": "__cpp_lib_transformation_trait_aliases",
   "values": {
     "c++14": int(201304),
   },
   "headers": ["type_traits"]
  },
  {"name": "__cpp_lib_result_of_sfinae",
   "values": {
     "c++14": int(201210),
   },
   "headers": ["functional", "type_traits"]
  },
  {"name": "__cpp_lib_is_final",
   "values": {
     "c++14": int(201402),
   },
   "headers": ["type_traits"]
  },
  {"name": "__cpp_lib_is_null_pointer",
   "values": {
     "c++14": int(201309),
   },
   "headers": ["type_traits"]
  },
  {"name": "__cpp_lib_chrono_udls",
   "values": {
     "c++14": int(201304),
   },
   "headers": ["chrono"]
  },
  {"name": "__cpp_lib_string_udls",
   "values": {
     "c++14": int(201304),
   },
   "headers": ["string"]
  },
  {"name": "__cpp_lib_generic_associative_lookup",
   "values": {
     "c++14": int(201304),
   },
   "headers": ["map", "set"]
  },
  {"name": "__cpp_lib_null_iterators",
   "values": {
     "c++14": int(201304),
   },
   "headers": ["iterator"]
  },
  {"name": "__cpp_lib_make_reverse_iterator",
   "values": {
     "c++14": int(201402),
   },
   "headers": ["iterator"]
  },
  {"name": "__cpp_lib_robust_nonmodifying_seq_ops",
   "values": {
     "c++14": int(201304),
   },
   "headers": ["algorithm"]
  },
  {"name": "__cpp_lib_complex_udls",
   "values": {
     "c++14": int(201309),
   },
   "headers": ["complex"]
  },
  {"name": "__cpp_lib_quoted_string_io",
   "values": {
     "c++14": int(201304),
   },
   "headers": ["iomanip"]
  },
  {"name": "__cpp_lib_shared_timed_mutex",
   "values": {
     "c++14": int(201402),
   },
   "headers": ["shared_mutex"],
   "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
   "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
  },
  # C++17 macros
  {"name": "__cpp_lib_atomic_is_always_lock_free",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["atomic"],
   "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
   "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
  },
  {"name": "__cpp_lib_filesystem",
   "values": {
     "c++17": int(201703),
   },
   "headers": ["filesystem"]
  },
  {"name": "__cpp_lib_invoke",
   "values": {
     "c++17": int(201411),
   },
   "headers": ["functional"]
  },
  {"name": "__cpp_lib_void_t",
   "values": {
     "c++17": int(201411),
   },
   "headers": ["type_traits"]
  },
  {"name": "__cpp_lib_node_extract",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["map", "set", "unordered_map", "unordered_set"]
  },
  {"name": "__cpp_lib_byte",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["cstddef"],
   },
  {"name": "__cpp_lib_hardware_interference_size",
   "values": {
     "c++17": int(201703),
   },
   "headers": ["new"],
   },
  {"name": "__cpp_lib_launder",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["new"],
   },
  {"name": "__cpp_lib_uncaught_exceptions",
   "values": {
     "c++17": int(201411),
   },
   "headers": ["exception"],
   },
  {"name": "__cpp_lib_as_const",
   "values": {
     "c++17": int(201510),
   },
   "headers": ["utility"],
   },
  {"name": "__cpp_lib_make_from_tuple",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["tuple"],
   },
  {"name": "__cpp_lib_apply",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["tuple"],
   },
  {"name": "__cpp_lib_optional",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["optional"],
   },
  {"name": "__cpp_lib_variant",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["variant"],
   },
  {"name": "__cpp_lib_any",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["any"],
   },
  {"name": "__cpp_lib_addressof_constexpr",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["memory"],
   "depends": "TEST_HAS_BUILTIN(__builtin_addressof) || TEST_GCC_VER >= 700",
   "internal_depends": "!defined(_LIBCPP_HAS_NO_BUILTIN_ADDRESSOF)",
   },
  {"name": "__cpp_lib_raw_memory_algorithms",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["memory"],
   },
  {"name": "__cpp_lib_enable_shared_from_this",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["memory"],
   },
  {"name": "__cpp_lib_shared_ptr_weak_type",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["memory"],
   },
  {"name": "__cpp_lib_shared_ptr_arrays",
   "values": {
     "c++17": int(201611),
   },
   "headers": ["memory"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_memory_resource",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["memory_resource"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_boyer_moore_searcher",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["functional"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_not_fn",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["functional"],
   },
  {"name": "__cpp_lib_bool_constant",
   "values": {
     "c++17": int(201505),
   },
   "headers": ["type_traits"],
   },
  {"name": "__cpp_lib_type_trait_variable_templates",
   "values": {
     "c++17": int(201510),
   },
   "headers": ["type_traits"],
   },
  {"name": "__cpp_lib_logical_traits",
   "values": {
     "c++17": int(201510),
   },
   "headers": ["type_traits"],
   },
  {"name": "__cpp_lib_is_swappable",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["type_traits"],
   },
  {"name": "__cpp_lib_is_invocable",
   "values": {
     "c++17": int(201703),
   },
   "headers": ["type_traits"],
   },
  {"name": "__cpp_lib_has_unique_object_representations",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["type_traits"],
   "depends": "TEST_HAS_BUILTIN_IDENTIFIER(__has_unique_object_representations) || TEST_GCC_VER >= 700",
   "internal_depends": "defined(_LIBCPP_HAS_UNIQUE_OBJECT_REPRESENTATIONS)",
   },
  {"name": "__cpp_lib_is_aggregate",
   "values": {
     "c++17": int(201703),
   },
   "headers": ["type_traits"],
   "depends": "TEST_HAS_BUILTIN_IDENTIFIER(__is_aggregate) || TEST_GCC_VER_NEW >= 7001",
   "internal_depends": "!defined(_LIBCPP_HAS_NO_IS_AGGREGATE)",
   },
  {"name": "__cpp_lib_chrono",
   "values": {
     "c++17": int(201611),
   },
   "headers": ["chrono"],
   },
  {"name": "__cpp_lib_execution",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["execution"],
   "unimplemented": True
   },
  {"name": "__cpp_lib_parallel_algorithm",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["algorithm", "numeric"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_to_chars",
   "values": {
     "c++17": int(201611),
   },
   "headers": ["utility"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_string_view",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["string", "string_view"],
   },
  {"name": "__cpp_lib_allocator_traits_is_always_equal",
   "values": {
     "c++17": int(201411),
   },
   "headers": ["memory", "scoped_allocator", "string", "deque", "forward_list", "list", "vector", "map", "set", "unordered_map", "unordered_set"],
   },
  {"name": "__cpp_lib_incomplete_container_elements",
   "values": {
     "c++17": int(201505),
   },
   "headers": ["forward_list", "list", "vector"],
   },
  {"name": "__cpp_lib_map_try_emplace",
   "values": {
     "c++17": int(201411),
   },
   "headers": ["map"],
   },
  {"name": "__cpp_lib_unordered_map_try_emplace",
   "values": {
     "c++17": int(201411),
   },
   "headers": ["unordered_map"],
   },
  {"name": "__cpp_lib_array_constexpr",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["iterator", "array"],
   },
  {"name": "__cpp_lib_nonmember_container_access",
   "values": {
     "c++17": int(201411),
   },
   "headers": ["iterator", "array", "deque", "forward_list", "list", "map", "regex",
               "set", "string", "unordered_map", "unordered_set", "vector"],
   },
  {"name": "__cpp_lib_sample",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["algorithm"],
   },
  {"name": "__cpp_lib_clamp",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["algorithm"],
   },
  {"name": "__cpp_lib_gcd_lcm",
   "values": {
     "c++17": int(201606),
   },
   "headers": ["numeric"],
   },
  {"name": "__cpp_lib_hypot",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["cmath"],
   },
  {"name": "__cpp_lib_math_special_functions",
   "values": {
     "c++17": int(201603),
   },
   "headers": ["cmath"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_shared_mutex",
   "values": {
     "c++17": int(201505),
   },
   "headers": ["shared_mutex"],
   "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
   "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
   },
  {"name": "__cpp_lib_scoped_lock",
   "values": {
     "c++17": int(201703),
   },
   "headers": ["mutex"],
   },
  # C++2a
  {"name": "__cpp_lib_char8_t",
   "values": {
     "c++2a": int(201811),
   },
   "headers": ["atomic", "filesystem", "istream", "limits", "locale", "ostream",
               "string", "string_view"],
   "depends": "defined(__cpp_char8_t)",
   "internal_depends": "!defined(_LIBCPP_NO_HAS_CHAR8_T)",
   },
  {"name": "__cpp_lib_erase_if",
   "values": {
     "c++2a": int(201811),
   },
   "headers": ["string", "deque", "forward_list", "list", "vector", "map",
               "set", "unordered_map", "unordered_set"]
  },
  {"name": "__cpp_lib_destroying_delete",
   "values": {
     "c++2a": int(201806),
   },
   "headers": ["new"],
   "depends":
      "TEST_STD_VER > 17"
      " && defined(__cpp_impl_destroying_delete)"
      " && __cpp_impl_destroying_delete >= 201806L",
   "internal_depends":
      "_LIBCPP_STD_VER > 17"
      " && defined(__cpp_impl_destroying_delete)"
      " && __cpp_impl_destroying_delete >= 201806L",
   },
  {"name": "__cpp_lib_three_way_comparison",
   "values": {
     "c++2a": int(201711),
   },
   "headers": ["compare"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_concepts",
   "values": {
     "c++2a": int(201806),
   },
   "headers": ["concepts"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_constexpr_swap_algorithms",
   "values": {
     "c++2a": int(201806),
   },
   "headers": ["algorithm"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_constexpr_misc",
   "values": {
     "c++2a": int(201811),
   },
   "headers": ["array", "functional", "iterator", "string_view", "tuple", "utility"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_bind_front",
   "values": {
     "c++2a": int(201811),
   },
   "headers": ["functional"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_is_constant_evaluated",
   "values": {
     "c++2a": int(201811),
   },
   "headers": ["type_traits"],
   "depends": "TEST_HAS_BUILTIN(__builtin_is_constant_evaluated) || TEST_GCC_VER >= 900",
   "internal_depends": "!defined(_LIBCPP_HAS_NO_BUILTIN_IS_CONSTANT_EVALUATED)",
   },
  {"name": "__cpp_lib_list_remove_return_type",
   "values": {
     "c++2a": int(201806),
   },
   "headers": ["forward_list", "list"],
   },
  {"name": "__cpp_lib_generic_unordered_lookup",
   "values": {
     "c++2a": int(201811),
   },
   "headers": ["unordered_map", "unordered_set"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_ranges",
   "values": {
     "c++2a": int(201811),
   },
   "headers": ["algorithm", "functional", "iterator", "memory", "ranges"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_bit_cast",
   "values": {
     "c++2a": int(201806),
   },
   "headers": ["bit"],
   "unimplemented": True,
   },
  {"name": "__cpp_lib_atomic_ref",
   "values": {
     "c++2a": int(201806),
   },
   "headers": ["atomic"],
   "unimplemented": True,
   "depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
   "internal_depends": "!defined(_LIBCPP_HAS_NO_THREADS)",
   },
  {"name": "__cpp_lib_interpolate",
   "values": {
     "c++2a": int(201902),
   },
   "headers": ["numeric"],
   },
  {"name": "__cpp_lib_endian",
   "values": {
     "c++2a": int(201907),
   },
   "headers": ["bit"],
   },
  {"name": "__cpp_lib_to_array",
   "values": {
     "c++2a": 201907L,
   },
   "headers": ["array"],
   },
]], key=lambda tc: tc["name"])

def get_std_dialects():
  std_dialects = ['c++14', 'c++17', 'c++2a']
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
  return result

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

#if _LIBCPP_STD_VER > 11
{cxx14_macros}
#endif

#if _LIBCPP_STD_VER > 14
{cxx17_macros}
#endif

#if _LIBCPP_STD_VER > 17
{cxx2a_macros}
#endif

#endif // _LIBCPP_VERSIONH
"""
  return template.format(
      synopsis=produce_version_synopsis().strip(),
      cxx14_macros=produce_macros_definition_for_std('c++14').strip(),
      cxx17_macros=produce_macros_definition_for_std('c++17').strip(),
      cxx2a_macros=produce_macros_definition_for_std('c++2a').strip())

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

  "defined":"""
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
  return result

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

def is_threading_header_unsafe_to_include(h):
  # NOTE: "<mutex>" does not blow up when included without threads.
  return h in ['atomic', 'shared_mutex']

def produce_tests():
  headers = set([h for tc in feature_test_macros for h in tc["headers"]])
  for h in headers:
    test_list = [tc for tc in feature_test_macros if h in tc["headers"]]
    if not has_header(h):
      for tc in test_list:
        assert 'unimplemented' in tc.keys()
      continue
    test_tags = ""
    if is_threading_header_unsafe_to_include(h):
      test_tags += '\n// UNSUPPORTED: libcpp-has-no-threads\n'
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
{test_tags}
// <{header}>

// Test the feature test macros defined by <{header}>

{synopsis}

#include <{header}>
#include "test_macros.h"

#if TEST_STD_VER < 14

{cxx11_tests}

#elif TEST_STD_VER == 14

{cxx14_tests}

#elif TEST_STD_VER == 17

{cxx17_tests}

#elif TEST_STD_VER > 17

{cxx2a_tests}

#endif // TEST_STD_VER > 17

int main(int, char**) {{ return 0; }}
""".format(script_name=script_name,
           header=h,
           test_tags=test_tags,
           synopsis=generate_synopsis(test_list),
           cxx11_tests=generate_std_test(test_list, 'c++11').strip(),
           cxx14_tests=generate_std_test(test_list, 'c++14').strip(),
           cxx17_tests=generate_std_test(test_list, 'c++17').strip(),
           cxx2a_tests=generate_std_test(test_list, 'c++2a').strip())
    test_name = "{header}.version.pass.cpp".format(header=h)
    out_path = os.path.join(macro_test_path, test_name)
    with open(out_path, 'w') as f:
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
  num_cols = len(grid[0])
  result = indent_str + add_divider(col_widths, 2)
  header_flag = 2
  for row_i in range(0, len(grid)):
    row = grid[row_i]
    result = result + indent_str + ' '.join([pad_cell(row[i], col_widths[i]) for i in range(0, len(row))]) + '\n'
    is_cxx_header = row[0].startswith('**')
    if row_i == len(grid) - 1:
      header_flag = 2
    result = result + indent_str + add_divider(col_widths, 1 if is_cxx_header else header_flag)
    header_flag = 0
  return result

def add_divider(widths, header_flag):
  if header_flag == 2:
    return ' '.join(['='*w for w in widths]) + '\n'
  if header_flag == 1:
    return '-'.join(['-'*w for w in widths]) + '\n'
  else:
    return ' '.join(['-'*w for w in widths]) + '\n'

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
  with open(table_doc_path, 'w') as f:
    f.write(doc_str)

def main():
  with tempfile.NamedTemporaryFile(mode='w', prefix='version.', delete=False) as tmp_file:
    print("producing new <version> header as %s" % tmp_file.name)
    tmp_file.write(produce_version_header())
  produce_tests()
  produce_docs()


if __name__ == '__main__':
  main()
