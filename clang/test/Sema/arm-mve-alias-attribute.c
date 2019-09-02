// RUN: %clang_cc1 -triple armv8.1m.main-arm-none-eabi -verify -fsyntax-only %s

static __inline__ __attribute__((__clang_arm_mve_alias(__builtin_arm_nop))) // expected-error {{'__clang_arm_mve_alias' attribute can only be applied to an ARM MVE builtin}}
void nop(void);

static __inline__ __attribute__((__clang_arm_mve_alias)) // expected-error {{'__clang_arm_mve_alias' attribute takes one argument}}
void noparens(void);

static __inline__ __attribute__((__clang_arm_mve_alias())) // expected-error {{'__clang_arm_mve_alias' attribute takes one argument}}
void emptyparens(void);

static __inline__ __attribute__((__clang_arm_mve_alias("string literal"))) // expected-error {{'__clang_arm_mve_alias' attribute requires parameter 1 to be an identifier}}
void stringliteral(void);

static __inline__ __attribute__((__clang_arm_mve_alias(1))) // expected-error {{'__clang_arm_mve_alias' attribute requires parameter 1 to be an identifier}}
void integer(void);

static __inline__ __attribute__((__clang_arm_mve_alias(__builtin_arm_nop, 2))) // expected-error {{'__clang_arm_mve_alias' attribute takes one argument}}
void twoargs(void);

static __attribute__((__clang_arm_mve_alias(__builtin_arm_nop))) // expected-error {{'__clang_arm_mve_alias' attribute only applies to functions}}
int variable;
