#include <stdarg.h>
#include <ptrcheck.h>

#pragma clang system_header

typedef void * (*variable_length_function)(va_list args);
static inline void* call_func_internal(variable_length_function f, va_list args) {
    return f(args);
}

static inline void* call_func(variable_length_function f, ...) {
    va_list ap;

    va_start(ap, f);
    return call_func_internal(f, ap);
}

