// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -emit-llvm -o /dev/null -verify=base %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -Wno-psabi -emit-llvm -o /dev/null -verify=avx %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx2 -Wno-psabi -emit-llvm -o /dev/null -verify=avx2 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -Wno-psabi -emit-llvm -o /dev/null -verify=avx512 %s

typedef float v8f __attribute__((vector_size(32)));
typedef float v16f __attribute__((vector_size(64)));

struct Wrapped256 { v8f value; };
struct Nested256 { Wrapped256 value; };
struct Array256 { v8f value[1]; };
struct Wrapped512 { v16f value; };
struct TwoVectors { v8f first, second; };
struct NonTrivial256 {
  v8f value;
  ~NonTrivial256();
};

__attribute__((target("default"))) int target_arg(v8f x) { return (int)x[0]; }
__attribute__((target("avx"))) int target_arg(v8f x) { return (int)x[0]; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}
int use_target_arg(v8f x) { return target_arg(x); }

__attribute__((target("default"))) int wrapped_arg(Wrapped256 x) { return (int)x.value[0]; }
__attribute__((target("avx"))) int wrapped_arg(Wrapped256 x) { return (int)x.value[0]; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}
int use_wrapped_arg(Wrapped256 x) { return wrapped_arg(x); }

__attribute__((target("default"))) int nested_arg(Nested256 x) { return (int)x.value.value[0]; }
__attribute__((target("avx"))) int nested_arg(Nested256 x) { return (int)x.value.value[0]; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}
int use_nested_arg(Nested256 x) { return nested_arg(x); }

__attribute__((target_clones("default", "avx")))
int array_arg(Array256 x) { return (int)x.value[0][0]; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}

__attribute__((target("default"))) int target_avx2_arg(v8f x) { return (int)x[0]; }
__attribute__((target("avx2"))) int target_avx2_arg(v8f x) { return (int)x[0]; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}
int use_target_avx2_arg(v8f x) { return target_avx2_arg(x); }

__attribute__((target_clones("default", "avx")))
int clone_arg(v8f x) { return (int)x[0]; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}

__attribute__((target("default"))) int target_wide(v16f x) { return (int)x[0]; }
__attribute__((target("avx512f"))) int target_wide(v16f x) { return (int)x[0]; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}
// avx-error@-2 {{target ABI is incompatible with the module ABI}}
// avx2-error@-3 {{target ABI is incompatible with the module ABI}}
int use_target_wide(v16f x) { return target_wide(x); }

__attribute__((target("default"))) v8f target_return(v8f x) { return x; }
__attribute__((target("avx"))) v8f target_return(v8f x) { return x; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}
v8f use_target_return(v8f x) { return target_return(x); }

__attribute__((target("default"))) Wrapped256 wrapped_return(Wrapped256 x) { return x; }
__attribute__((target("avx"))) Wrapped256 wrapped_return(Wrapped256 x) { return x; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}
Wrapped256 use_wrapped_return(Wrapped256 x) { return wrapped_return(x); }

__attribute__((target("default"))) Wrapped512 wide_return(Wrapped512 x) { return x; }
__attribute__((target("avx512f"))) Wrapped512 wide_return(Wrapped512 x) { return x; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}
// avx-error@-2 {{target ABI is incompatible with the module ABI}}
// avx2-error@-3 {{target ABI is incompatible with the module ABI}}
Wrapped512 use_wide_return(Wrapped512 x) { return wide_return(x); }

__attribute__((target("default"))) v8f target_avx2_return(v8f x) { return x; }
__attribute__((target("avx2"))) v8f target_avx2_return(v8f x) { return x; }
// base-error@-1 {{target ABI is incompatible with the module ABI}}
v8f use_target_avx2_return(v8f x) { return target_avx2_return(x); }

// A 512-bit vector is passed in memory by both versions below.
__attribute__((target("default"))) int memory_arg(v16f x) { return (int)x[0]; }
__attribute__((target("avx"))) int memory_arg(v16f x) { return (int)x[0]; }

// A record with two vectors is passed in memory by every version.
__attribute__((target("default"))) int two_vectors(TwoVectors x) { return (int)x.first[0]; }
__attribute__((target("avx512f"))) int two_vectors(TwoVectors x) { return (int)x.first[0]; }

// A nontrivial C++ record is passed indirectly at every AVX level.
__attribute__((target("default"))) int nontrivial_arg(NonTrivial256 x) { return (int)x.value[0]; }
__attribute__((target("avx"))) int nontrivial_arg(NonTrivial256 x) { return (int)x.value[0]; }

__attribute__((target_clones("default", "avx")))
int scalar_arg(int x) { return x; }

// avx512-no-diagnostics
