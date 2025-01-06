# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import pathlib, functools

libcxx_root = pathlib.Path(__file__).resolve().parent.parent.parent
libcxx_include = libcxx_root / "include"
assert libcxx_root.exists()

def _is_header_file(file):
    """Returns whether the given file is a header file, i.e. not a directory or the modulemap file."""
    return not file.is_dir() and not file.name in [
        "module.modulemap",
        "CMakeLists.txt",
        "libcxx.imp",
        "__config_site.in",
    ]

@functools.total_ordering
class Header:
    _name: str
    """Relative path from the root of libcxx/include"""

    def __init__(self, name: str):
        """Create a Header.

        name: The path of the header relative to libc++'s include directory.
              For example '__algorithm/find.h' or 'coroutine'.
        """
        self._name = name

    def is_public(self) -> bool:
        """Returns whether the header is a public libc++ API header."""
        return "__" not in self._name and not self._name.startswith("ext/")

    def is_internal(self) -> bool:
        """Returns whether the header is an internal implementation detail of the library."""
        return not self.is_public()

    def is_C_compatibility(self) -> bool:
        """
        Returns whether the header is a C compatibility header (headers ending in .h like stdlib.h).

        Note that headers like <cstdlib> are not considered C compatibility headers.
        """
        return self.is_public() and self._name.endswith(".h")

    def is_cstd(self) -> bool:
        """Returns whether the header is a C 'std' header, like <cstddef>, <cerrno>, etc."""
        return self._name in [
            "cassert",
            "ccomplex",
            "cctype",
            "cerrno",
            "cfenv",
            "cfloat",
            "cinttypes",
            "ciso646",
            "climits",
            "clocale",
            "cmath",
            "csetjmp",
            "csignal",
            "cstdalign",
            "cstdarg",
            "cstdbool",
            "cstddef",
            "cstdint",
            "cstdio",
            "cstdlib",
            "cstring",
            "ctgmath",
            "ctime",
            "cuchar",
            "cwchar",
            "cwctype",
        ]

    def is_experimental(self) -> bool:
        """Returns whether the header is a public experimental header."""
        return self.is_public() and self._name.startswith("experimental/")

    def has_cxx20_module(self) -> bool:
        """
        Returns whether the header is in the std and std.compat C++20 modules.

        These headers are all C++23-and-later headers, excluding C compatibility headers and
        experimental headers.
        """
        # These headers have been removed in C++20 so are never part of a module.
        removed_in_20 = ["ccomplex", "ciso646", "cstdalign", "cstdbool", "ctgmath"]
        return self.is_public() and not self.is_experimental() and not self.is_C_compatibility() and not self._name in removed_in_20

    def is_cxx03_frozen_header(self) -> bool:
        """Returns whether the header is a frozen C++03 support header."""
        return self._name.startswith("__cxx03/")

    def is_in_modulemap(self) -> bool:
        """Returns whether a header should be listed in the modulemap."""
        # TODO: Should `__config_site` be in the modulemap?
        if self._name == "__config_site":
            return False

        if self._name == "__assertion_handler":
            return False

        # exclude libc++abi files
        if self._name in ["cxxabi.h", "__cxxabi_config.h"]:
            return False

        # exclude headers in __support/ - these aren't supposed to work everywhere,
        # so they shouldn't be included in general
        if self._name.startswith("__support/"):
            return False

        # exclude ext/ headers - these are non-standard extensions and are barely
        # maintained. People should migrate away from these and we don't need to
        # burden ourself with maintaining them in any way.
        if self._name.startswith("ext/"):
            return False

        # TODO: Frozen C++03 headers should probably be in the modulemap as well
        if self.is_cxx03_frozen_header():
            return False

        return True

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return repr(self._name)

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self._name == other
        return self._name == other._name

    def __lt__(self, other) -> bool:
        if isinstance(other, str):
            return self._name < other
        return self._name < other._name

    def __hash__(self) -> int:
        return hash(self._name)


# Commonly-used sets of headers
all_headers = [Header(p.relative_to(libcxx_include).as_posix()) for p in libcxx_include.rglob("[_a-z]*") if _is_header_file(p)]
all_headers += [Header("__config_site"), Header("__assertion_handler")] # Headers generated during the build process
public_headers = [h for h in all_headers if h.is_public()]
module_headers = [h for h in all_headers if h.has_cxx20_module()]
module_c_headers = [h for h in all_headers if h.has_cxx20_module() and h.is_cstd()]

# These headers are not yet implemented in libc++
#
# These headers are required by the latest (draft) Standard but have not been
# implemented yet. They are used in the generated module input. The C++23 standard
# modules will fail to build if a header is added but this list is not updated.
headers_not_available = list(map(Header, [
    "debugging",
    "flat_set",
    "generator",
    "hazard_pointer",
    "inplace_vector",
    "linalg",
    "rcu",
    "spanstream",
    "stacktrace",
    "stdfloat",
    "text_encoding",
]))

header_restrictions = {
    # headers with #error directives
    "atomic": "_LIBCPP_HAS_ATOMIC_HEADER",
    "stdatomic.h": "_LIBCPP_HAS_ATOMIC_HEADER",

    # headers with #error directives
    "ios": "_LIBCPP_HAS_LOCALIZATION",
    # transitive includers of the above headers
    "clocale": "_LIBCPP_HAS_LOCALIZATION",
    "codecvt": "_LIBCPP_HAS_LOCALIZATION",
    "fstream": "_LIBCPP_HAS_LOCALIZATION",
    "iomanip": "_LIBCPP_HAS_LOCALIZATION",
    "iostream": "_LIBCPP_HAS_LOCALIZATION",
    "istream": "_LIBCPP_HAS_LOCALIZATION",
    "locale": "_LIBCPP_HAS_LOCALIZATION",
    "ostream": "_LIBCPP_HAS_LOCALIZATION",
    "regex": "_LIBCPP_HAS_LOCALIZATION",
    "sstream": "_LIBCPP_HAS_LOCALIZATION",
    "streambuf": "_LIBCPP_HAS_LOCALIZATION",
    "strstream": "_LIBCPP_HAS_LOCALIZATION",
    "syncstream": "_LIBCPP_HAS_LOCALIZATION",
}

lit_header_restrictions = {
    "barrier": "// UNSUPPORTED: no-threads, c++03, c++11, c++14, c++17",
    "clocale": "// UNSUPPORTED: no-localization",
    "codecvt": "// UNSUPPORTED: no-localization",
    "coroutine": "// UNSUPPORTED: c++03, c++11, c++14, c++17",
    "cwchar": "// UNSUPPORTED: no-wide-characters",
    "cwctype": "// UNSUPPORTED: no-wide-characters",
    "experimental/iterator": "// UNSUPPORTED: c++03",
    "experimental/propagate_const": "// UNSUPPORTED: c++03",
    "experimental/simd": "// UNSUPPORTED: c++03",
    "experimental/type_traits": "// UNSUPPORTED: c++03",
    "experimental/utility": "// UNSUPPORTED: c++03",
    "filesystem": "// UNSUPPORTED: no-filesystem, c++03, c++11, c++14",
    "fstream": "// UNSUPPORTED: no-localization, no-filesystem",
    "future": "// UNSUPPORTED: no-threads, c++03",
    "iomanip": "// UNSUPPORTED: no-localization",
    "ios": "// UNSUPPORTED: no-localization",
    "iostream": "// UNSUPPORTED: no-localization",
    "istream": "// UNSUPPORTED: no-localization",
    "latch": "// UNSUPPORTED: no-threads, c++03, c++11, c++14, c++17",
    "locale": "// UNSUPPORTED: no-localization",
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
    "syncstream": "// UNSUPPORTED: no-localization",
    "thread": "// UNSUPPORTED: no-threads, c++03",
    "wchar.h": "// UNSUPPORTED: no-wide-characters",
    "wctype.h": "// UNSUPPORTED: no-wide-characters",
}

# Undeprecate headers that are deprecated in C++17 and removed in C++20.
lit_header_undeprecations = {
    "ccomplex": "// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS",
    "ciso646": "// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS",
    "cstdalign": "// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS",
    "cstdbool": "// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS",
    "ctgmath": "// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS",
}

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
    "flat_map": ["compare", "initializer_list"],
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
    "syncstream": ["ostream"],
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
