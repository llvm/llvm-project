// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs %s -verify

// RUN: rm -rf %t.pch.cache
// RUN: %clang_cc1 -fmodules-cache-path=%t.pch.cache -fmodules -fimplicit-module-maps -I %S/Inputs -emit-pch -o %t.pch -x objective-c-header %S/Inputs/use-builtin.h
// RUN: %clang_cc1 -fmodules-cache-path=%t.pch.cache -fmodules -fimplicit-module-maps -I %S/Inputs %s -include-pch %t.pch %s -verify

// expected-no-diagnostics

void use_constant_string_builtins1(void) {
  (void)__builtin___CFStringMakeConstantString("");
  (void)__builtin___NSStringMakeConstantString("");
}

@import builtin;

int foo() {
  return __builtin_object_size(p, 0);
}

@import builtin.sub;

int bar() {
  return __builtin_object_size(p, 0);
}

int baz() {
  return IS_CONST(0);
}

void use_constant_string_builtins2(void) {
  (void)__builtin___CFStringMakeConstantString("");
  (void)__builtin___NSStringMakeConstantString("");
}
