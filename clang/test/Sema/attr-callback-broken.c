// RUN: %clang_cc1 %s -verify -fsyntax-only

__attribute__((callback())) void no_callee(void (*callback)(void)); // expected-error {{'callback' attribute specifies no callback callee}}

__attribute__((callback(1, 1))) void too_many_args_1(void (*callback)(void)) {}      // expected-error {{'callback' attribute takes one argument}}
__attribute__((callback(1, -1))) void too_many_args_2(double (*callback)(void));     // expected-error {{'callback' attribute takes one argument}}
__attribute__((callback(1, 2, 2))) void too_many_args_3(void (*callback)(int), int); // expected-error {{'callback' attribute requires exactly 2 arguments}}

__attribute__((callback(1, 2))) void too_few_args_1(void (*callback)(int, int), int); // expected-error {{'callback' attribute takes one argument}}
__attribute__((callback(1))) void too_few_args_2(int (*callback)(int));               // expected-error {{'callback' attribute takes no arguments}}
__attribute__((callback(1, -1))) void too_few_args_3(void (*callback)(int, int)) {}   // expected-error {{'callback' attribute takes one argument}}

__attribute__((callback(-1))) void oob_args_1(void (*callback)(void));         // expected-error {{'callback' attribute specifies invalid callback callee}}
__attribute__((callback(2))) void oob_args_2(int *(*callback)(void)) {}        // expected-error {{'callback' attribute parameter 1 is out of bounds}}
__attribute__((callback(1, 3))) void oob_args_3(short (*callback)(int), int);  // expected-error {{'callback' attribute parameter 2 is out of bounds}}
__attribute__((callback(-2, 2))) void oob_args_4(void *(*callback)(int), int); // expected-error {{'callback' attribute parameter 1 is out of bounds}}
__attribute__((callback(1, -2))) void oob_args_5(void *(*callback)(int), int); // expected-error {{'callback' attribute parameter 2 is out of bounds}}
__attribute__((callback(1, 2))) void oob_args_6(void *(*callback)(int), ...);  // expected-error {{'callback' attribute parameter 2 is out of bounds}}

__attribute__((callback(1))) __attribute__((callback(1))) void multiple_cb_1(void (*callback)(void));                           // expected-error {{multiple 'callback' attributes specified}}
__attribute__((callback(1))) __attribute__((callback(2))) void multiple_cb_2(void (*callback1)(void), void (*callback2)(void)); // expected-error {{multiple 'callback' attributes specified}}

#ifdef HAS_THIS
__attribute__((callback(0))) void oob_args_0(void (*callback)(void)); // expected-error {{'callback' attribute specifies invalid callback callee}}
#else
__attribute__((callback(0))) void oob_args_0(void (*callback)(void));                 // expected-error {{'callback' argument at position 1 references unavailable implicit 'this'}}
__attribute__((callback(1, 0))) void no_this_1(void *(*callback)(void *));            // expected-error {{'callback' argument at position 2 references unavailable implicit 'this'}}
__attribute__((callback(1, 0))) void no_this_2(void *(*callback)(int, void *));       // expected-error {{'callback' argument at position 2 references unavailable implicit 'this'}}
#endif

// We could allow the following declarations if we at some point need to:

__attribute__((callback(1, -1))) void vararg_cb_1(void (*callback)(int, ...)) {}     // expected-error {{'callback' attribute callee may not be variadic}}
__attribute__((callback(1, 1))) void vararg_cb_2(void (*callback)(int, ...), int a); // expected-error {{'callback' attribute callee may not be variadic}}

__attribute__((callback(1, -1, 1, 2, 3, 4, -1))) void varargs_1(void (*callback)(int, ...), int a, float b, double c) {}               // expected-error {{'callback' attribute requires exactly 6 arguments}}
__attribute__((callback(1, -1, 4, 2, 3, 4, -1))) void varargs_2(void (*callback)(void *, double, int, ...), int a, float b, double c); // expected-error {{'callback' attribute requires exactly 6 arguments}}

__attribute__((callback(1, -1, 1))) void self_arg_1(void (*callback)(int, ...)) {}          // expected-error {{'callback' attribute requires exactly 2 arguments}}
__attribute__((callback(1, -1, 1, -1, -1, 1))) void self_arg_2(void (*callback)(int, ...)); // expected-error {{'callback' attribute requires exactly 5 arguments}}

__attribute__((callback(cb))) void unknown_name1(void (*callback)(void)) {}     // expected-error {{'callback' attribute argument 'cb' is not a known function parameter}}
__attribute__((callback(cb, ab))) void unknown_name2(void (*cb)(int), int a) {} // expected-error {{'callback' attribute argument 'ab' is not a known function parameter}}

__attribute__((callback(callback, 1))) void too_many_args_1b(void (*callback)(void)) {}      // expected-error {{'callback' attribute takes one argument}}
__attribute__((callback(callback, __))) void too_many_args_2b(double (*callback)(void));     // expected-error {{'callback' attribute takes one argument}}
__attribute__((callback(callback, 2, 2))) void too_many_args_3b(void (*callback)(int), int); // expected-error {{'callback' attribute requires exactly 2 arguments}}

__attribute__((callback(callback, a))) void too_few_args_1b(void (*callback)(int, int), int a); // expected-error {{'callback' attribute takes one argument}}
__attribute__((callback(callback))) void too_few_args_2b(int (*callback)(int));                 // expected-error {{'callback' attribute takes no arguments}}
__attribute__((callback(callback, __))) void too_few_args_3b(void (*callback)(int, int)) {}     // expected-error {{'callback' attribute takes one argument}}

__attribute__((callback(__))) void oob_args_1b(void (*callback)(void)); // expected-error {{'callback' attribute specifies invalid callback callee}}

__attribute__((callback(callback))) __attribute__((callback(callback))) void multiple_cb_1b(void (*callback)(void));                     // expected-error {{multiple 'callback' attributes specified}}
__attribute__((callback(1))) __attribute__((callback(callback2))) void multiple_cb_2b(void (*callback1)(void), void (*callback2)(void)); // expected-error {{multiple 'callback' attributes specified}}

#ifdef HAS_THIS
__attribute__((callback(this))) void oob_args_0b(void (*callback)(void)); // expected-error {{'callback' attribute specifies invalid callback callee}}
#else
__attribute__((callback(this))) void oob_args_0b(void (*callback)(void));           // expected-error {{'callback' argument at position 1 references unavailable implicit 'this'}}
__attribute__((callback(1, this))) void no_this_1b(void *(*callback)(void *));      // expected-error {{'callback' argument at position 2 references unavailable implicit 'this'}}
__attribute__((callback(1, this))) void no_this_2b(void *(*callback)(int, void *)); // expected-error {{'callback' argument at position 2 references unavailable implicit 'this'}}
#endif

// We could allow the following declarations if we at some point need to:

__attribute__((callback(callback, __))) void vararg_cb_1b(void (*callback)(int, ...)) {} // expected-error {{'callback' attribute callee may not be variadic}}
__attribute__((callback(1, a))) void vararg_cb_2b(void (*callback)(int, ...), int a);    // expected-error {{'callback' attribute callee may not be variadic}}

__attribute__((callback(callback, __, callback, a, b, c, __))) void varargs_1b(void (*callback)(int, ...), int a, float b, double c) {} // expected-error {{'callback' attribute requires exactly 6 arguments}}
__attribute__((callback(1, __, c, a, b, c, -1))) void varargs_2b(void (*callback)(void *, double, int, ...), int a, float b, double c); // expected-error {{'callback' attribute requires exactly 6 arguments}}

__attribute__((callback(1, __, callback))) void self_arg_1b(void (*callback)(int, ...)) {}                        // expected-error {{'callback' attribute requires exactly 2 arguments}}
__attribute__((callback(callback, __, callback, __, __, callback))) void self_arg_2b(void (*callback)(int, ...)); // expected-error {{'callback' attribute requires exactly 5 arguments}}
