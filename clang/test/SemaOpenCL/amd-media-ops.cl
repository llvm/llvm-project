// RUN: %clang_cc1 %s -triple amdgcn-unknown-unknown -cl-ext=-all,+cl_amd_media_ops,+cl_amd_media_ops2 -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL -finclude-default-header
// RUN: %clang_cc1 %s -triple amdgcn-unknown-unknown -cl-ext=-all,+cl_amd_media_ops,+cl_amd_media_ops2 -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL -finclude-default-header -fdeclare-opencl-builtins
// expected-no-diagnostics
// RUN: %clang_cc1 %s -triple amdgcn-unknown-unknown -cl-ext=-all,+cl_amd_media_ops2 -verify=ops  -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL -finclude-default-header
// RUN: %clang_cc1 %s -triple amdgcn-unknown-unknown -cl-ext=-all,+cl_amd_media_ops  -verify=ops2 -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL -finclude-default-header
// RUN: %clang_cc1 %s -triple amdgcn-unknown-unknown -cl-ext=-all,+cl_amd_media_ops2 -verify=ops  -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL -finclude-default-header -fdeclare-opencl-builtins
// RUN: %clang_cc1 %s -triple amdgcn-unknown-unknown -cl-ext=-all,+cl_amd_media_ops  -verify=ops2 -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL -finclude-default-header -fdeclare-opencl-builtins

#define TEST_1ARG_BUILTIN_WITH_TYPE(builtin, ret, type) \
    ret test_ ## builtin ## _ ## ret ## _## type (type a) { \
        return builtin(a); \
    }

#define TEST_2ARG_BUILTIN_WITH_TYPE(builtin, ret, type) \
    ret test_ ## builtin ## _ ## ret ## _## type (type a, type b) { \
        return builtin(a, b); \
    }

#define TEST_3ARG_BUILTIN_WITH_TYPE(builtin, ret, type) \
    ret test_ ## builtin ## _ ## ret ## _ ## type (type a, type b, type c) { \
        return builtin(a, b, c); \
    }

#define TEST_1ARG_BUILTIN(builtin, ret, type) \
    TEST_1ARG_BUILTIN_WITH_TYPE(builtin, ret, type) \
    TEST_1ARG_BUILTIN_WITH_TYPE(builtin, ret ## 2, type ## 2) \
    TEST_1ARG_BUILTIN_WITH_TYPE(builtin, ret ## 3, type ## 3) \
    TEST_1ARG_BUILTIN_WITH_TYPE(builtin, ret ## 4, type ## 4) \
    TEST_1ARG_BUILTIN_WITH_TYPE(builtin, ret ## 8, type ## 8) \
    TEST_1ARG_BUILTIN_WITH_TYPE(builtin, ret ## 16, type ## 16)

#define TEST_2ARG_BUILTIN(builtin, ret, type) \
    TEST_2ARG_BUILTIN_WITH_TYPE(builtin, ret, type) \
    TEST_2ARG_BUILTIN_WITH_TYPE(builtin, ret ## 2, type ## 2) \
    TEST_2ARG_BUILTIN_WITH_TYPE(builtin, ret ## 3, type ## 3) \
    TEST_2ARG_BUILTIN_WITH_TYPE(builtin, ret ## 4, type ## 4) \
    TEST_2ARG_BUILTIN_WITH_TYPE(builtin, ret ## 8, type ## 8) \
    TEST_2ARG_BUILTIN_WITH_TYPE(builtin, ret ## 16, type ## 16)

#define TEST_3ARG_BUILTIN(builtin, ret, type) \
    TEST_3ARG_BUILTIN_WITH_TYPE(builtin, ret, type) \
    TEST_3ARG_BUILTIN_WITH_TYPE(builtin, ret ## 2, type ## 2) \
    TEST_3ARG_BUILTIN_WITH_TYPE(builtin, ret ## 3, type ## 3) \
    TEST_3ARG_BUILTIN_WITH_TYPE(builtin, ret ## 4, type ## 4) \
    TEST_3ARG_BUILTIN_WITH_TYPE(builtin, ret ## 8, type ## 8) \
    TEST_3ARG_BUILTIN_WITH_TYPE(builtin, ret ## 16, type ## 16)

#define TEST_3ARG_BUILTIN_WITH_TYPES(builtin, ret, type_a, type_b, type_c) \
    ret test_ ## builtin ## _ ## ret ## _ ## type_a ## _ ## type_b ## _ ## type_c (type_a a, type_b b, type_c c) { \
        return builtin(a, b, c); \
    }

#define TEST_3ARG_WITH_TYPES_BUILTIN(builtin, ret, type_a, type_b, type_c) \
    TEST_3ARG_BUILTIN_WITH_TYPES(builtin, ret, type_a, type_b, type_c) \
    TEST_3ARG_BUILTIN_WITH_TYPES(builtin, ret ## 2, type_a ## 2, type_b ## 2, type_c ## 2) \
    TEST_3ARG_BUILTIN_WITH_TYPES(builtin, ret ## 3, type_a ## 3, type_b ## 3, type_c ## 3) \
    TEST_3ARG_BUILTIN_WITH_TYPES(builtin, ret ## 4, type_a ## 4, type_b ## 4, type_c ## 4) \
    TEST_3ARG_BUILTIN_WITH_TYPES(builtin, ret ## 8, type_a ## 8, type_b ## 8, type_c ## 8) \
    TEST_3ARG_BUILTIN_WITH_TYPES(builtin, ret ## 16, type_a ## 16, type_b ## 16, type_c ## 16)

TEST_3ARG_BUILTIN(amd_bitalign, uint, uint) // ops-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_bytealign, uint, uint) // ops-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_lerp, uint, uint) // ops-error 6 {{use of undeclared identifier}}

uint test_amd_pack(float4 a) {
    return amd_pack(a); // ops-error{{use of undeclared identifier}}
}

uint test_amd_sad4(uint4 a, uint4 b, uint c) {
    return amd_sad4(a, b, c); // ops-error{{use of undeclared identifier}}
}

TEST_3ARG_BUILTIN(amd_sadhi, uint, uint)// ops-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_sad, uint, uint) // ops-error 6 {{use of undeclared identifier}}


TEST_1ARG_BUILTIN(amd_unpack0, float, uint) // ops-error 6 {{use of undeclared identifier}}
TEST_1ARG_BUILTIN(amd_unpack1, float, uint) // ops-error 6 {{use of undeclared identifier}}
TEST_1ARG_BUILTIN(amd_unpack2, float, uint) // ops-error 6 {{use of undeclared identifier}}
TEST_1ARG_BUILTIN(amd_unpack3, float, uint) // ops-error 6 {{use of undeclared identifier}}

TEST_3ARG_WITH_TYPES_BUILTIN(amd_bfe, int, int, uint, uint) // ops2-error 6 {{use of undeclared identifier}} #amd_bfe0
TEST_3ARG_BUILTIN(amd_bfe, uint, uint) // ops2-error 6 {{use of undeclared identifier}} #amd_bfe1
TEST_2ARG_BUILTIN(amd_bfm, uint, uint) // ops2-error 6 {{use of undeclared identifier}}

TEST_3ARG_BUILTIN(amd_max3, float, float) // ops2-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_max3, int, int) // ops2-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_max3, uint, uint) // ops2-error 6 {{use of undeclared identifier}}

TEST_3ARG_BUILTIN(amd_median3, float, float) // ops2-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_median3, int, int) // ops2-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_median3, uint, uint) // ops2-error 6 {{use of undeclared identifier}}

TEST_3ARG_BUILTIN(amd_min3, float, float) // ops2-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_min3, int, int) // ops2-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_min3, uint, uint) // ops2-error 6 {{use of undeclared identifier}}

TEST_3ARG_WITH_TYPES_BUILTIN(amd_mqsad, ulong, ulong, uint, ulong) // ops2-error 6 {{use of undeclared identifier}}
TEST_3ARG_WITH_TYPES_BUILTIN(amd_qsad, ulong, ulong, uint, ulong) // ops2-error 6 {{use of undeclared identifier}}

TEST_3ARG_BUILTIN(amd_msad, uint, uint) // ops2-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_sadd, uint, uint) // ops2-error 6 {{use of undeclared identifier}}
TEST_3ARG_BUILTIN(amd_sadw, uint, uint) // ops2-error 6 {{use of undeclared identifier}}
