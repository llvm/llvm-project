// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=c,expected -DNO_JMP_BUF %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=c,expected -DWRONG_JMP_BUF %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=c,expected -DRIGHT_JMP_BUF %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=c,expected -DONLY_JMP_BUF %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=c,expected -DNO_SETJMP %s -ast-dump 2>&1 | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=cxx,expected -x c++ -DNO_JMP_BUF %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=cxx,expected -x c++ -DWRONG_JMP_BUF %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=cxx,expected -x c++ -DRIGHT_JMP_BUF %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=cxx,expected -x c++ -DONLY_JMP_BUF %s -ast-dump | FileCheck %s --check-prefixes=CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=cxx,expected -x c++ -DNO_SETJMP %s -ast-dump | FileCheck %s --check-prefixes=CHECK2

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NO_JMP_BUF
// This happens in some versions of glibc: the declaration of __sigsetjmp
// precedes the declaration of sigjmp_buf.
extern long setjmp(long *); // Can't check, so we trust that this is the right type
// FIXME: We could still diagnose the missing `jmp_buf` at the point of the call.
// c-no-diagnostics
#elif WRONG_JMP_BUF
typedef long jmp_buf;
// FIXME: Consider producing a similar warning in C++.
extern int setjmp(char); // c-warning {{incompatible redeclaration of library function 'setjmp'}}
                         // c-note@-1 {{'setjmp' is a builtin with type 'int (jmp_buf)' (aka 'int (long)')}}
#elif RIGHT_JMP_BUF
typedef long jmp_buf;
extern int setjmp(long); // OK, right type.
#elif ONLY_JMP_BUF
typedef int *jmp_buf;
#endif

void use(void) {
  setjmp(0);
  #if NO_SETJMP
  // cxx-error@-2 {{undeclared identifier 'setjmp'}}
  // c-warning@-3 {{implicit declaration of function 'setjmp' is invalid in C99}}
  #elif ONLY_JMP_BUF
  // cxx-error@-5 {{undeclared identifier 'setjmp'}}
  // c-warning@-6 {{implicitly declaring library function 'setjmp' with type 'int (jmp_buf)' (aka 'int (int *)')}}
  // c-note@-7 {{include the header <setjmp.h> or explicitly provide a declaration for 'setjmp'}}
  #else
  // cxx-no-diagnostics
  #endif

  #ifdef NO_SETJMP
  // In this case, the regular AST dump doesn't dump the implicit declaration of 'setjmp'.
  #pragma clang __debug dump setjmp
  #endif
}

// CHECK1: FunctionDecl {{.*}} used setjmp
// CHECK1: BuiltinAttr {{.*}} Implicit
// CHECK1: ReturnsTwiceAttr {{.*}} Implicit

// mingw declares _setjmp with an unusual signature.
int _setjmp(void *, void *);
#if !defined(NO_JMP_BUF) && !defined(NO_SETJMP)
// c-warning@-2 {{incompatible redeclaration of library function '_setjmp'}}
// c-note@-3 {{'_setjmp' is a builtin with type 'int (jmp_buf)'}}
#endif
void use_mingw(void) {
  _setjmp(0, 0);
}

// CHECK2: FunctionDecl {{.*}} used _setjmp
// CHECK2: BuiltinAttr {{.*}} Implicit
// CHECK2: ReturnsTwiceAttr {{.*}} Implicit

#ifdef __cplusplus
}
#endif
