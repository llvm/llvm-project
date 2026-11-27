// RUN: %clang_cc1 -fsyntax-only -verify %s

int __attribute__((stack_protector_ignore)) global_var; // expected-warning {{'stack_protector_ignore' attribute only applies to local variables}}

void __attribute__((stack_protector_ignore)) func(void) {} // expected-warning {{'stack_protector_ignore' attribute only applies to local variables}}

void func2(void) {
	__attribute__((stack_protector_ignore)) int var;
	__attribute__((stack_protector_ignore)) static int var2; // expected-warning {{'stack_protector_ignore' attribute only applies to local variables}}
	__attribute__((stack_protector_ignore(2))) int var3; // expected-error {{'stack_protector_ignore' attribute takes no arguments}}
}
