// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -std=c++20 -triple amdgcn -target-cpu tahiti -emit-llvm -fcuda-is-device -verify=no-memrealtime -o - %s
// RUN: %clang_cc1 -std=c++20 -triple amdgcn -target-cpu gfx950 -emit-llvm -fcuda-is-device -o - %s

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))

struct S {
    static constexpr auto memrealtime_lambda = []() {
        __builtin_amdgcn_s_memrealtime(); // no-memrealtime-error{{'__builtin_amdgcn_s_memrealtime' needs target feature s-memrealtime}}
    };
};

__attribute__((target("s-memrealtime")))
__device__ void test_target_dependant_builtin_attr_fail() {
    S::memrealtime_lambda();
}

constexpr auto memrealtime_lambda = []() {
    __builtin_amdgcn_s_memrealtime(); // no-memrealtime-error{{'__builtin_amdgcn_s_memrealtime' needs target feature s-memrealtime}}
};

__attribute__((target("s-memrealtime")))
__device__ void global_test_target_dependant_builtin_attr_fail() {
    memrealtime_lambda();
}

__attribute__((target("s-memrealtime")))
__device__ void local_test_target_dependant_builtin_attr_fail() {
    static constexpr auto f = []() {
        __builtin_amdgcn_s_memrealtime(); // no-memrealtime-error{{'__builtin_amdgcn_s_memrealtime' needs target feature s-memrealtime}}
    };
    f();
}
