/* 6.9 Preprocessor Directives and Macros
 * Some of these are handled by clang or passed by clover */
#if __OPENCL_VERSION__ >= 110
#define CLC_VERSION_1_0 100
#define CLC_VERSION_1_1 110
#endif

#if __OPENCL_VERSION__ >= 120
#define CLC_VERSION_1_2 120
#endif

#define NULL ((void*)0)

#define __kernel_exec(X, typen) __kernel \
                                __attribute__((work_group_size_hint(X, 1, 1))) \
                                __attribute__((vec_type_hint(typen)))

#define kernel_exec(X, typen) __kernel_exec(X, typen)
