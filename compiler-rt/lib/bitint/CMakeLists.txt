add_compiler_rt_component(bitint)

include(builtin-config-ix)

add_compiler_rt_runtime(clang_rt.bitint
    STATIC
    ARCHS ${BITINT_SUPPORTED_ARCH}
    SOURCES ../builtins/udivmodei5.c
    CFLAGS ${SANITIZER_COMMON_CFLAGS} -DIS_BITINT_LIBRARY
    PARENT_TARGET bitint)

