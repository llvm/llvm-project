// Counts on parameters and on callback return types survive serialization,
// still naming the parameters they were parsed against, and redeclarations
// compare against them.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -x c-header -emit-pch -o %t/params.pch %t/params.h
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -include-pch %t/params.pch -fsyntax-only -verify %t/use.c
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -include-pch %t/params.pch -ast-dump-all %t/use.c | FileCheck %s

// A header with errors still leaves no empty count to trip deserialization.
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -x c-header -emit-pch -fallow-pcm-with-compiler-errors -verify -o %t/errors.pch %t/errors.h
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -include-pch %t/errors.pch -fallow-pcm-with-compiler-errors -ast-dump-all %t/use-errors.c | FileCheck %s --check-prefix=ERRORS

//--- params.h
#define __counted_by(f)  __attribute__((counted_by(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))

int fwd(int *__counted_by(count) buf, int count);
int array(int arr[] __sized_by(size), int size);
void cb_return(int *__counted_by(len) (*cb)(int len));
void cb_param(void (*cb)(int *__counted_by(m) p, int m));
void out(int *__counted_by(*len) *buf, int *len);
struct callbacks {
  int len;
  void *__sized_by(len) (*ret)(int len);
};

//--- use.c
// expected-no-diagnostics
void out(int *__counted_by(*size) *buf, int *size);

int use(int *p, struct callbacks *c) {
  return fwd(p, 4) + array(p, 16) + (c->ret != 0);
}

// CHECK: FunctionDecl {{.*}} imported {{.*}}fwd 'int (int * __counted_by(count), int)'
// CHECK: FunctionDecl {{.*}} imported {{.*}}array 'int (int * __sized_by(size), int)'
// CHECK: FunctionDecl {{.*}} imported {{.*}}cb_return 'void (int * __counted_by(len)(*)(int))'
// CHECK: FunctionDecl {{.*}} imported {{.*}}cb_param 'void (void (*)(int * __counted_by(m), int))'
// CHECK: FunctionDecl {{.*}} imported {{.*}}out 'void (int * __counted_by(*len)*, int *)'
// CHECK: FieldDecl {{.*}} imported {{.*}}ret 'void * __sized_by(len)(*)(int)'

//--- errors.h
#define __counted_by(f)  __attribute__((counted_by(f)))
int g;
void rejected(float n, int *__counted_by(n) p); // expected-error {{'counted_by' requires a non-boolean integer type argument}}
void global(int *__counted_by(g) p); // expected-error {{count expression in function declaration may only reference function parameters}}
void unparseable(int *__counted_by(+) p, int n); // expected-error {{expected expression}}
void fn_typed(int *__counted_by(n) cb(int n)); // expected-error {{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'int * __counted_by(n)(int)' (aka 'int *(int)') is a function type}}

//--- use-errors.c
int main(void) { return 0; }

// ERRORS: FunctionDecl {{.*}} invalid rejected 'void (float, int * __counted_by(n))'
// ERRORS: FunctionDecl {{.*}} invalid global 'void (int * __counted_by(g))'
// ERRORS: FunctionDecl {{.*}} invalid unparseable 'void (int * __counted_by(<recovery-expr>()), int)'
// ERRORS: FunctionDecl {{.*}} invalid fn_typed 'void (int * __counted_by(n)(*)(int))'
