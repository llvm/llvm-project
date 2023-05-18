// RUN: %clang_cc1 %s -verify -fsyntax-only

// expected-no-diagnostics

__attribute__((callback(1))) void no_args(void (*callback)(void));
__attribute__((callback(1, 2, 3))) void args_1(void (*callback)(int, double), int a, double b);
__attribute__((callback(2, 3, 3))) void args_2(int a, void (*callback)(double, double), double b);
__attribute__((callback(2, -1, -1))) void args_3(int a, void (*callback)(double, double), double b);

__attribute__((callback(callback))) void no_argsb(void (*callback)(void));
__attribute__((callback(callback, a, 3))) void args_1b(void (*callback)(int, double), int a, double b);
__attribute__((callback(callback, b, b))) void args_2b(int a, void (*callback)(double, double), double b);
__attribute__((callback(2, __, __))) void args_3b(int a, void (*callback)(double, double), double b);
__attribute__((callback(callback, -1, __))) void args_3c(int a, void (*callback)(double, double), double b);
