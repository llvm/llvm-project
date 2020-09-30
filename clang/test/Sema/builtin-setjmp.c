// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify -DNO_JMP_BUF %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify -DWRONG_JMP_BUF %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify -DRIGHT_JMP_BUF %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify -DONLY_JMP_BUF %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify -DNO_SETJMP %s -ast-dump 2>&1 | FileCheck %s

#ifdef NO_JMP_BUF
// This happens in some versions of glibc: the declaration of __sigsetjmp
// precedes the declaration of sigjmp_buf.
extern long setjmp(long *); // Can't check, so we trust that this is the right type
// FIXME: We could still diagnose the missing `jmp_buf` at the point of the call.
// expected-no-diagnostics
#elif WRONG_JMP_BUF
typedef long jmp_buf;
extern int setjmp(char); // expected-warning {{incompatible redeclaration of library function 'setjmp'}}
                         // expected-note@-1 {{'setjmp' is a builtin with type 'int (jmp_buf)' (aka 'int (long)')}}
#elif RIGHT_JMP_BUF
typedef long jmp_buf;
extern int setjmp(long); // OK, right type.
// expected-no-diagnostics
#elif ONLY_JMP_BUF
typedef int *jmp_buf;
#endif

void use() {
  setjmp(0);
  #ifdef NO_SETJMP
  // expected-warning@-2 {{implicit declaration of function 'setjmp' is invalid in C99}}
  #elif ONLY_JMP_BUF
  // expected-warning@-4 {{implicitly declaring library function 'setjmp' with type 'int (jmp_buf)' (aka 'int (int *)')}}
  // expected-note@-5 {{include the header <setjmp.h> or explicitly provide a declaration for 'setjmp'}}
  #endif

  #ifdef NO_SETJMP
  // In this case, the regular AST dump doesn't dump the implicit declaration of 'setjmp'.
  #pragma clang __debug dump setjmp
  #endif
}

// CHECK: FunctionDecl {{.*}} used setjmp
// CHECK: BuiltinAttr {{.*}} Implicit
// CHECK: ReturnsTwiceAttr {{.*}} Implicit
