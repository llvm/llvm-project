// RUN: %clang_cc1 -stack-protector 2 -Rpass-missed=inline -O2 -verify %s -emit-llvm-only

void side_effect(void);

void foo(void) {
  side_effect();
}

// expected-remark@+3 {{foo will not be inlined into bar: stack protected callee but caller requested no stack protector}}
__attribute__((no_stack_protector))
void bar(void) {
  foo();
}

// expected-remark@+2 {{bar will not be inlined into baz: stack protected caller but callee requested no stack protector}}
void baz(void) {
  bar();
}

void ssp_callee(void);

// No issue; matching stack protections.
void ssp_caller(void) {
  ssp_callee();
}

__attribute__((no_stack_protector))
void nossp_callee(void);

// No issue; matching stack protections.
__attribute__((no_stack_protector))
void nossp_caller(void) {
  nossp_callee();
}
