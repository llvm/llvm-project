// RUN: %clang_cc1 -fsyntax-only %s

typedef struct foo T0;
typedef const struct foo T1;

int a0[__builtin_types_compatible_p(T0,
                                    const T1) ? 1 : -1];
