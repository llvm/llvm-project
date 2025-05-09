// RUN: %clang_cc1 -std=c++20 -triple amdgcn -target-cpu gfx90a -fsyntax-only -fcuda-is-device -verify=gfx90a -o - %s
// RUN: %clang_cc1 -std=c++20 -triple amdgcn -target-cpu gfx950 -fsyntax-only -fcuda-is-device -o - %s

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))

struct S {
    static constexpr auto make_buffer_rsrc_lambda = [](void *p, short stride, int num, int flags) {
        return __builtin_amdgcn_make_buffer_rsrc(p, stride, num, flags);
    };

    static constexpr auto global_load_lds_lambda = [](void* src, __shared__ void *dst) {
        __builtin_amdgcn_global_load_lds(src, dst, 16, 0, 0); // gfx90a-error{{invalid size value}} gfx90a-note{{size must be 1, 2, or 4}}
    };
};

__device__ __amdgpu_buffer_rsrc_t test_simple_builtin(void *p, short stride, int num, int flags) {
    return S::make_buffer_rsrc_lambda(p, stride, num, flags);
}

__device__ void test_target_dependant_builtin(void *src, __shared__ void *dst) {
    S::global_load_lds_lambda(src, dst);
}

constexpr auto make_buffer_rsrc_lambda = [](void *p, short stride, int num, int flags) {
    return __builtin_amdgcn_make_buffer_rsrc(p, stride, num, flags);
};

constexpr auto global_load_lds_lambda = [](void* src, __shared__ void *dst) {
    __builtin_amdgcn_global_load_lds(src, dst, 16, 0, 0); // gfx90a-error{{invalid size value}} gfx90a-note{{size must be 1, 2, or 4}}
};

__device__ __amdgpu_buffer_rsrc_t global_test_simple_builtin(void *p, short stride, int num, int flags) {
    return make_buffer_rsrc_lambda(p, stride, num, flags);
}

__device__ void global_test_target_dependant_builtin(void *src, __shared__ void *dst) {
    global_load_lds_lambda(src, dst);
}

__device__ __amdgpu_buffer_rsrc_t local_test_simple_builtin(void *p, short stride, int num, int flags) {
    constexpr auto f = [](void *p, short stride, int num, int flags) {
        return __builtin_amdgcn_make_buffer_rsrc(p, stride, num, flags);
    };
    return f(p, stride, num, flags);
}

__device__ void local_test_target_dependant_builtin(void *src, __shared__ void *dst) {
    constexpr auto f = [](void* src, __shared__ void *dst) {
        __builtin_amdgcn_global_load_lds(src, dst, 16, 0, 0); // gfx90a-error{{invalid size value}} gfx90a-note{{size must be 1, 2, or 4}}
    };
    f(src, dst);
}
