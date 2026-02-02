# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from libcxx.test.dsl import Feature, compilerMacros, programSucceeds

features = []

# Deduce and add the test features that that are implied by the #defines in
# the <__config> header.
#
# For each macro of the form `_LIBCPP_XXX_YYY_ZZZ` defined below that
# is defined after including <__config>, add a Lit feature called
# `libcpp-xxx-yyy-zzz`. When a macro is defined to a specific value
# (e.g. `_LIBCPP_ABI_VERSION=2`), the feature is `libcpp-xxx-yyy-zzz=<value>`.
#
# Note that features that are more strongly tied to libc++ are named libcpp-foo,
# while features that are more general in nature are not prefixed with 'libcpp-'.
macros = {
    "_LIBCPP_NO_VCRUNTIME": "libcpp-no-vcruntime",
    "_LIBCPP_ABI_VERSION": "libcpp-abi-version",
    "_LIBCPP_ABI_BOUNDED_ITERATORS": "libcpp-has-abi-bounded-iterators",
    "_LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL": "libcpp-has-abi-bounded-iterators-in-optional",
    "_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STRING": "libcpp-has-abi-bounded-iterators-in-string",
    "_LIBCPP_ABI_BOUNDED_ITERATORS_IN_VECTOR": "libcpp-has-abi-bounded-iterators-in-vector",
    "_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STD_ARRAY": "libcpp-has-abi-bounded-iterators-in-std-array",
    "_LIBCPP_ABI_BOUNDED_UNIQUE_PTR": "libcpp-has-abi-bounded-unique_ptr",
    "_LIBCPP_ABI_FIX_UNORDERED_CONTAINER_SIZE_TYPE": "libcpp-has-abi-fix-unordered-container-size-type",
    "_LIBCPP_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR": "libcpp-deprecated-abi-disable-pair-trivial-copy-ctor",
    "_LIBCPP_ABI_NO_COMPRESSED_PAIR_PADDING": "libcpp-abi-no-compressed-pair-padding",
    "_LIBCPP_PSTL_BACKEND_LIBDISPATCH": "libcpp-pstl-backend-libdispatch",
}
for macro, feature in macros.items():
    features.append(
        Feature(
            name=lambda cfg, m=macro, f=feature: f + ("={}".format(compilerMacros(cfg)[m]) if compilerMacros(cfg)[m] else ""),
            when=lambda cfg, m=macro: m in compilerMacros(cfg),
        )
    )

true_false_macros = {
    "_LIBCPP_HAS_THREAD_API_EXTERNAL": "libcpp-has-thread-api-external",
    "_LIBCPP_HAS_THREAD_API_PTHREAD": "libcpp-has-thread-api-pthread",
}
for macro, feature in true_false_macros.items():
    features.append(
        Feature(
            name=feature,
            when=lambda cfg, m=macro: m in compilerMacros(cfg)
            and compilerMacros(cfg)[m] == "1",
        )
    )

inverted_macros = {
    "_LIBCPP_HAS_TIME_ZONE_DATABASE": "no-tzdb",
    "_LIBCPP_HAS_FILESYSTEM": "no-filesystem",
    "_LIBCPP_HAS_LOCALIZATION": "no-localization",
    "_LIBCPP_HAS_THREADS": "no-threads",
    "_LIBCPP_HAS_MONOTONIC_CLOCK": "no-monotonic-clock",
    "_LIBCPP_HAS_WIDE_CHARACTERS": "no-wide-characters",
    "_LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS": "libcpp-has-no-availability-markup",
    "_LIBCPP_HAS_RANDOM_DEVICE": "no-random-device",
    "_LIBCPP_HAS_UNICODE": "libcpp-has-no-unicode",
    "_LIBCPP_HAS_TERMINAL": "no-terminal",
}
for macro, feature in inverted_macros.items():
    features.append(
        Feature(
            name=feature,
            when=lambda cfg, m=macro: m in compilerMacros(cfg)
            and compilerMacros(cfg)[m] == "0",
        )
    )

for mode in ("none", "fast", "extensive", "debug"):
    check_program = f"""
        #include <stddef.h> // any header to get the definitions
        int main(int, char**) {{
        #if defined(_LIBCPP_VERSION) && \\
                defined(_LIBCPP_HARDENING_MODE) && _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_{mode.upper()}
            return 0;
        #else
            return 1;
        #endif
        }}
    """
    features.append(
        Feature(
            name=f"libcpp-hardening-mode={mode}",
            when=lambda cfg, prog=check_program: programSucceeds(cfg, prog)
        )
    )
