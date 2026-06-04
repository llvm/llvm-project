// REQUIRES: amdgpu-registered-target
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// Test with the pre-202604 stl
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %t/stl/include \
// RUN:   -internal-isystem %t/stl/2025/include \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -verify %t/main.hip -fcuda-is-device -o /dev/null
//
// Test with the 202604 stl
// RUN: not %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %t/stl/include \
// RUN:   -internal-isystem %t/stl/2026/include \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -verify %t/main.hip -fcuda-is-device -o /dev/null

#--- stl/include/stl_common
_CLANG_BUILTIN2(isunordered)
_CLANG_BUILTIN2(isgreater)
_CLANG_BUILTIN2(isgreaterequal)
_CLANG_BUILTIN2(isless)
_CLANG_BUILTIN2(islessequal)
_CLANG_BUILTIN2(islessgreater)

#define FP_NAN 0
#define FP_INFINITE 1
#define FP_ZERO 2
#define FP_SUBNORMAL 3
#define FP_NORMAL 4
#--- stl/2025/include/version
#define _MSVC_STL_UPDATE 202508L
#--- stl/2025/include/cmath
#define _CLANG_BUILTIN2(NAME) \
    bool NAME(float x, float y) noexcept { \
        return __builtin_##NAME(x, y); \
    }

#include <stl_common>
#--- stl/2026/include/version
#define _MSVC_STL_UPDATE 202604L
#--- stl/2026/include/cmath
#define _CLANG_BUILTIN2(NAME) \
    bool constexpr NAME(float x, float y) noexcept { \
        return __builtin_##NAME(x, y); \
    }

#include <stl_common>
#--- main.hip
// expected-no-diagnostics

#include <cmath>

#define TEST_CMP(NAME) \
    bool __attribute__((device)) test_device_ ## NAME (float x) { \
        bool b = NAME(x, x); \
        return b; \
    }

TEST_CMP(isunordered)
TEST_CMP(isgreater)
TEST_CMP(isgreaterequal)
TEST_CMP(isless)
TEST_CMP(islessequal)
TEST_CMP(islessgreater)
