# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from libcxx.test.dsl import compilerMacros, Feature, AddCompileFlag, AddFeature

_isAnyClang = lambda cfg: "__clang__" in compilerMacros(cfg)
_isAppleClang = lambda cfg: "__apple_build_version__" in compilerMacros(cfg)
_isAnyGCC = lambda cfg: "__GNUC__" in compilerMacros(cfg)
_isClang = lambda cfg: _isAnyClang(cfg) and not _isAppleClang(cfg)
_isGCC = lambda cfg: _isAnyGCC(cfg) and not _isAnyClang(cfg)
_isAnyClangOrGCC = lambda cfg: _isAnyClang(cfg) or _isAnyGCC(cfg)
_isClExe = lambda cfg: not _isAnyClangOrGCC(cfg)
_isMSVC = lambda cfg: "_MSC_VER" in compilerMacros(cfg)
_msvcVersion = lambda cfg: (int(compilerMacros(cfg)["_MSC_VER"]) // 100, int(compilerMacros(cfg)["_MSC_VER"]) % 100)

features = [
    # gcc-style-warnings detects compilers that understand -Wno-meow flags, unlike MSVC's compiler driver cl.exe.
    Feature(name="gcc-style-warnings", when=_isAnyClangOrGCC),
    Feature(name="cl-style-warnings", when=_isClExe),

    Feature(name="apple-clang", when=_isAppleClang),
    Feature(
        name=lambda cfg: "apple-clang-{__clang_major__}".format(**compilerMacros(cfg)),
        when=_isAppleClang,
    ),
    Feature(
        name=lambda cfg: "apple-clang-{__clang_major__}.{__clang_minor__}".format(**compilerMacros(cfg)),
        when=_isAppleClang,
    ),
    Feature(
        name=lambda cfg: "apple-clang-{__clang_major__}.{__clang_minor__}.{__clang_patchlevel__}".format(**compilerMacros(cfg)),
        when=_isAppleClang,
    ),
    Feature(name="clang", when=_isClang),
    Feature(
        name=lambda cfg: "clang-{__clang_major__}".format(**compilerMacros(cfg)),
        when=_isClang,
    ),
    Feature(
        name=lambda cfg: "clang-{__clang_major__}.{__clang_minor__}".format(**compilerMacros(cfg)),
        when=_isClang,
    ),
    Feature(
        name=lambda cfg: "clang-{__clang_major__}.{__clang_minor__}.{__clang_patchlevel__}".format(**compilerMacros(cfg)),
        when=_isClang,
    ),
    # Note: Due to a GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=104760), we must disable deprecation warnings
    #       on GCC or spurious diagnostics are issued.
    #
    # TODO:
    # - Enable -Wplacement-new with GCC.
    # - Enable -Wclass-memaccess with GCC.
    Feature(
        name="gcc",
        when=_isGCC,
        actions=[
            AddCompileFlag("-D_LIBCPP_DISABLE_DEPRECATION_WARNINGS"),
            AddCompileFlag("-Wno-placement-new"),
            AddCompileFlag("-Wno-class-memaccess"),
            AddFeature("GCC-ALWAYS_INLINE-FIXME"),
        ],
    ),
    Feature(
        name=lambda cfg: "gcc-{__GNUC__}".format(**compilerMacros(cfg)), when=_isGCC
    ),
    Feature(
        name=lambda cfg: "gcc-{__GNUC__}.{__GNUC_MINOR__}".format(**compilerMacros(cfg)),
        when=_isGCC,
    ),
    Feature(
        name=lambda cfg: "gcc-{__GNUC__}.{__GNUC_MINOR__}.{__GNUC_PATCHLEVEL__}".format(**compilerMacros(cfg)),
        when=_isGCC,
    ),
    Feature(name="msvc", when=_isMSVC),
    Feature(name=lambda cfg: "msvc-{}".format(*_msvcVersion(cfg)), when=_isMSVC),
    Feature(name=lambda cfg: "msvc-{}.{}".format(*_msvcVersion(cfg)), when=_isMSVC),
]
