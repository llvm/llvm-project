# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import os, pathlib

lit_header_restrictions = {
    "barrier": "// UNSUPPORTED: no-threads, c++03, c++11, c++14, c++17",
    "clocale": "// UNSUPPORTED: no-localization",
    "codecvt": "// UNSUPPORTED: no-localization",
    "coroutine": "// UNSUPPORTED: c++03, c++11, c++14, c++17",
    "cwchar": "// UNSUPPORTED: no-wide-characters",
    "cwctype": "// UNSUPPORTED: no-wide-characters",
    "experimental/algorithm": "// UNSUPPORTED: c++03",
    "experimental/deque": "// UNSUPPORTED: c++03",
    "experimental/forward_list": "// UNSUPPORTED: c++03",
    "experimental/functional": "// UNSUPPORTED: c++03",
    "experimental/iterator": "// UNSUPPORTED: c++03",
    "experimental/list": "// UNSUPPORTED: c++03",
    "experimental/map": "// UNSUPPORTED: c++03",
    "experimental/memory_resource": "// UNSUPPORTED: c++03",
    "experimental/propagate_const": "// UNSUPPORTED: c++03",
    "experimental/regex": "// UNSUPPORTED: no-localization, c++03",
    "experimental/set": "// UNSUPPORTED: c++03",
    "experimental/simd": "// UNSUPPORTED: c++03",
    "experimental/string": "// UNSUPPORTED: c++03",
    "experimental/type_traits": "// UNSUPPORTED: c++03",
    "experimental/unordered_map": "// UNSUPPORTED: c++03",
    "experimental/unordered_set": "// UNSUPPORTED: c++03",
    "experimental/utility": "// UNSUPPORTED: c++03",
    "experimental/vector": "// UNSUPPORTED: c++03",
    "filesystem": "// UNSUPPORTED: no-filesystem, c++03, c++11, c++14",
    "fstream": "// UNSUPPORTED: no-localization, no-filesystem",
    "future": "// UNSUPPORTED: no-threads, c++03",
    "iomanip": "// UNSUPPORTED: no-localization",
    "ios": "// UNSUPPORTED: no-localization",
    "iostream": "// UNSUPPORTED: no-localization",
    "istream": "// UNSUPPORTED: no-localization",
    "latch": "// UNSUPPORTED: no-threads, c++03, c++11, c++14, c++17",
    "locale": "// UNSUPPORTED: no-localization",
    "locale.h": "// UNSUPPORTED: no-localization",
    "mutex": "// UNSUPPORTED: no-threads, c++03",
    "ostream": "// UNSUPPORTED: no-localization",
    "print": "// UNSUPPORTED: no-filesystem, c++03, c++11, c++14, c++17, c++20, availability-fp_to_chars-missing", # TODO PRINT investigate
    "regex": "// UNSUPPORTED: no-localization",
    "semaphore": "// UNSUPPORTED: no-threads, c++03, c++11, c++14, c++17",
    "shared_mutex": "// UNSUPPORTED: no-threads, c++03, c++11",
    "sstream": "// UNSUPPORTED: no-localization",
    "stdatomic.h": "// UNSUPPORTED: no-threads, c++03, c++11, c++14, c++17, c++20",
    "stop_token": "// UNSUPPORTED: no-threads, c++03, c++11, c++14, c++17",
    "streambuf": "// UNSUPPORTED: no-localization",
    "strstream": "// UNSUPPORTED: no-localization",
    "thread": "// UNSUPPORTED: no-threads, c++03",
    "wchar.h": "// UNSUPPORTED: no-wide-characters",
    "wctype.h": "// UNSUPPORTED: no-wide-characters",
}

private_headers_still_public_in_modules = [
    "__assert",
    "__config",
    "__config_site.in",
    "__hash_table",
    "__threading_support",
    "__tree",
    "__undef_macros",
    "__verbose_abort",
]

# Headers that can't be included on their own. Most of these are conceptually
# part of another header that were split out just for organization, but aren't
# meant to be included by anything else.
non_standalone_headers = frozenset((
    # Alternate implementations for __algorithm/pstl_backends/cpu_backends/backend.h
    "__algorithm/pstl_backends/cpu_backends/serial.h",
    "__algorithm/pstl_backends/cpu_backends/thread.h",

    # Alternate implementations for locale.
    "__locale_dir/locale_base_api/bsd_locale_defaults.h",
    "__locale_dir/locale_base_api/bsd_locale_fallbacks.h",
))

# This table was produced manually, by grepping the TeX source of the Standard's
# library clauses for the string "#include". Each header's synopsis contains
# explicit "#include" directives for its mandatory inclusions.
# For example, [algorithm.syn] contains "#include <initializer_list>".
mandatory_inclusions = {
    "algorithm": ["initializer_list"],
    "array": ["compare", "initializer_list"],
    "bitset": ["iosfwd", "string"],
    "chrono": ["compare"],
    "cinttypes": ["cstdint"],
    "complex.h": ["complex"],
    "coroutine": ["compare"],
    "deque": ["compare", "initializer_list"],
    "filesystem": ["compare"],
    "forward_list": ["compare", "initializer_list"],
    "ios": ["iosfwd"],
    "iostream": ["ios", "istream", "ostream", "streambuf"],
    "iterator": ["compare", "concepts"],
    "list": ["compare", "initializer_list"],
    "map": ["compare", "initializer_list"],
    "memory": ["compare"],
    "optional": ["compare"],
    "queue": ["compare", "initializer_list"],
    "random": ["initializer_list"],
    "ranges": ["compare", "initializer_list", "iterator"],
    "regex": ["compare", "initializer_list"],
    "set": ["compare", "initializer_list"],
    "stack": ["compare", "initializer_list"],
    "string_view": ["compare"],
    "string": ["compare", "initializer_list"],
    # TODO "syncstream": ["ostream"],
    "system_error": ["compare"],
    "tgmath.h": ["cmath", "complex"],
    "thread": ["compare"],
    "tuple": ["compare"],
    "typeindex": ["compare"],
    "unordered_map": ["compare", "initializer_list"],
    "unordered_set": ["compare", "initializer_list"],
    "utility": ["compare", "initializer_list"],
    "valarray": ["initializer_list"],
    "variant": ["compare"],
    "vector": ["compare", "initializer_list"],
}

def is_header(file):
    """Returns whether the given file is a header (i.e. not a directory or the modulemap file)."""
    return (
        not file.is_dir()
        and not file.name == "module.modulemap.in"
        and not file.name == "CMakeLists.txt"
        and file.name != "libcxx.imp"
    )

libcxx_root = pathlib.Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
include = pathlib.Path(os.path.join(libcxx_root, "include"))
test = pathlib.Path(os.path.join(libcxx_root, "test"))
assert libcxx_root.exists()

toplevel_headers = sorted(
    p.relative_to(include).as_posix() for p in include.glob("[a-z]*") if is_header(p)
)
experimental_headers = sorted(
    p.relative_to(include).as_posix() for p in include.glob("experimental/[a-z]*") if is_header(p)
)
public_headers = toplevel_headers + experimental_headers
private_headers = sorted(
    p.relative_to(include).as_posix() for p in include.rglob("*") if is_header(p)
                                                                     and str(p.relative_to(include)).startswith("__")
                                                                     and not p.name.startswith("pstl")
                                                                     and str(p.relative_to(include)) not in non_standalone_headers
)
