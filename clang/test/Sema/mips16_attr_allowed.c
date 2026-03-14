// RUN: %clang_cc1 -triple mipsel-linux-gnu -fsyntax-only -verify %s

void foo32(void);
void foo16(void);
void __attribute__((nomips16)) foo32(void);
void __attribute__((mips16)) foo16(void);

void __attribute__((nomips16)) foo32_(void);
void __attribute__((mips16)) foo16_(void);
void foo32_(void);
void foo16_(void);

void foo32__(void) __attribute__((nomips16));
void foo32__(void) __attribute__((mips16));

void foo32a(void) __attribute__((nomips16(0))) ; // expected-error {{'nomips16' attribute takes no arguments}}
void __attribute__((mips16(1))) foo16a(void); // expected-error {{'mips16' attribute takes no arguments}}

void __attribute__((nomips16(1, 2))) foo32b(void); // expected-error {{'nomips16' attribute takes no arguments}}
void __attribute__((mips16(1, 2))) foo16b(void); // expected-error {{'mips16' attribute takes no arguments}}


__attribute((nomips16)) int a; // expected-error {{attribute only applies to functions}}

__attribute((mips16)) int b; // expected-error {{attribute only applies to functions}}


