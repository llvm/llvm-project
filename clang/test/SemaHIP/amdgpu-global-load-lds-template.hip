// RUN: %clang_cc1 -std=c++20 -triple amdgcn -target-cpu gfx950 -fsyntax-only -fcuda-is-device -verify %s

// Test that __builtin_amdgcn_global_load_lds and related builtins accept
// instantiation-dependent expressions (e.g., template parameters, sizeof(T))
// as the size argument. These should be accepted during template definition
// and checked during template instantiation.

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))

using size_t = decltype(sizeof(int));

template<int N>
__device__ void test_global_load_lds_template(void* src, __shared__ void* dst) {
    // Template parameter should be accepted as size argument
    __builtin_amdgcn_global_load_lds(src, dst, N, 0, 0); // #load_lds
}

template<int N>
__device__ void test_load_to_lds_template(void* src, __shared__ void* dst) {
    __builtin_amdgcn_load_to_lds(src, dst, N, 0, 0); // #load_to_lds
}

// Test with valid sizes - these should compile without errors
template __device__ void test_global_load_lds_template<1>(void*, __shared__ void*);
template __device__ void test_global_load_lds_template<2>(void*, __shared__ void*);
template __device__ void test_global_load_lds_template<4>(void*, __shared__ void*);
template __device__ void test_global_load_lds_template<12>(void*, __shared__ void*);
template __device__ void test_global_load_lds_template<16>(void*, __shared__ void*);

template __device__ void test_load_to_lds_template<1>(void*, __shared__ void*);
template __device__ void test_load_to_lds_template<2>(void*, __shared__ void*);
template __device__ void test_load_to_lds_template<4>(void*, __shared__ void*);
template __device__ void test_load_to_lds_template<12>(void*, __shared__ void*);
template __device__ void test_load_to_lds_template<16>(void*, __shared__ void*);

// Test with constexpr computed from template parameter
template<int N>
__device__ void test_computed_size(void* src, __shared__ void* dst) {
    constexpr int Size = N * 2;
    __builtin_amdgcn_global_load_lds(src, dst, Size, 0, 0); // #computed
}

template __device__ void test_computed_size<2>(void*, __shared__ void*);  // Size = 4
template __device__ void test_computed_size<8>(void*, __shared__ void*);  // Size = 16

// Test that invalid sizes are caught at instantiation time
template __device__ void test_global_load_lds_template<5>(void*, __shared__ void*); // #inst1
// expected-error@#load_lds {{invalid size value}}
// expected-note@#load_lds {{size must be 1, 2, 4, 12 or 16}}
// expected-note@#inst1 {{in instantiation of function template specialization 'test_global_load_lds_template<5>' requested here}}

template __device__ void test_load_to_lds_template<7>(void*, __shared__ void*); // #inst2
// expected-error@#load_to_lds {{invalid size value}}
// expected-note@#load_to_lds {{size must be 1, 2, 4, 12 or 16}}
// expected-note@#inst2 {{in instantiation of function template specialization 'test_load_to_lds_template<7>' requested here}}

template __device__ void test_computed_size<3>(void*, __shared__ void*);  // #inst3
// expected-error@#computed {{invalid size value}}
// expected-note@#computed {{size must be 1, 2, 4, 12 or 16}}
// expected-note@#inst3 {{in instantiation of function template specialization 'test_computed_size<3>' requested here}}

// Test with sizeof(T) - this is instantiation-dependent but NOT value-dependent.
// The type of sizeof is always size_t, but the value depends on T.
template<typename T>
__device__ void test_sizeof_type(void* src, __shared__ void* dst) {
    __builtin_amdgcn_global_load_lds(src, dst, sizeof(T), 0, 0); // #sizeof_type
}

// Valid: sizeof(char) = 1, sizeof(short) = 2, sizeof(int) = 4
template __device__ void test_sizeof_type<char>(void*, __shared__ void*);
template __device__ void test_sizeof_type<short>(void*, __shared__ void*);
template __device__ void test_sizeof_type<int>(void*, __shared__ void*);

// Invalid: sizeof(double) = 8 (not a valid size)
struct Eight { char x[8]; };
template __device__ void test_sizeof_type<Eight>(void*, __shared__ void*); // #inst_sizeof
// expected-error@#sizeof_type {{invalid size value}}
// expected-note@#sizeof_type {{size must be 1, 2, 4, 12 or 16}}
// expected-note@#inst_sizeof {{in instantiation of function template specialization 'test_sizeof_type<Eight>' requested here}}

// Test with expression involving both type and non-type template parameters
template<typename T, int N>
__device__ void test_mixed_dependent(void* src, __shared__ void* dst) {
    __builtin_amdgcn_global_load_lds(src, dst, sizeof(T) * N, 0, 0); // #mixed
}

// Valid: sizeof(short) * 2 = 4
template __device__ void test_mixed_dependent<short, 2>(void*, __shared__ void*);
// Valid: sizeof(int) * 4 = 16
template __device__ void test_mixed_dependent<int, 4>(void*, __shared__ void*);

// Invalid: sizeof(int) * 2 = 8
template __device__ void test_mixed_dependent<int, 2>(void*, __shared__ void*); // #inst_mixed
// expected-error@#mixed {{invalid size value}}
// expected-note@#mixed {{size must be 1, 2, 4, 12 or 16}}
// expected-note@#inst_mixed {{in instantiation of function template specialization 'test_mixed_dependent<int, 2>' requested here}}
