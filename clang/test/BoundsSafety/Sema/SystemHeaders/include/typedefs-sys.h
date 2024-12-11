#include <ptrcheck.h>

typedef const char * __null_terminated my_str_explicit_t;
typedef const char * my_str_implicit_t;
typedef int * __null_terminated my_nt_int_ptr_t;
typedef int * my_int_ptr_t;

// both-error@+1{{'__counted_by' inside typedef is only allowed for function type}}
typedef int * __counted_by(4) ivec4_t; // If this is ever allowed we need to handle it for system headers.

#pragma clang system_header

static inline my_str_implicit_t funcInSDK1(const char *p) {
    my_str_implicit_t str = p;
    return str;
    return p;
}

static inline my_str_explicit_t funcInSDK2(const char *p) {
    //strict-error@+1{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char *' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    my_str_explicit_t str = p;
    return str;
    //strict-error@+1{{returning 'const char *' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    return p;
}

static inline my_nt_int_ptr_t funcInSDK3(int *p) {
    //strict-error@+1{{initializing 'int *__single __terminated_by(0)' (aka 'int *__single') with an expression of incompatible type 'int *' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    my_nt_int_ptr_t p2 = p;
    return p2;
    //strict-error@+1{{returning 'int *' from a function with incompatible result type 'int *__single __terminated_by(0)' (aka 'int *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    return p;
}

typedef my_int_ptr_t __null_terminated nt_local_t;

static inline nt_local_t funcInSDK4(int *p) {
    //strict-error@+1{{initializing 'int *__single __terminated_by(0)' (aka 'int *__single') with an expression of incompatible type 'int *' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    nt_local_t p2 = p;
    return p2;
    //strict-error@+1{{returning 'int *' from a function with incompatible result type 'int *__single __terminated_by(0)' (aka 'int *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    return p;
}

