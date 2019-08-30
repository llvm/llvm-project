// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -Wno-private-extern -verify -pedantic %s -DGNU
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -Wno-private-extern -verify -pedantic -x c++ %s -DGNU -std=c++98
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -Wno-private-extern -verify -pedantic %s -std=c11 -DC11 -D__thread=_Thread_local
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -Wno-private-extern -verify=expected,thread-local -pedantic -x c++ %s -DC11 -D__thread=_Thread_local -std=c++98
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -Wno-private-extern -verify -pedantic -x c++ %s -DCXX11 -D__thread=thread_local -std=c++11 -Wno-deprecated
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -Wno-private-extern -verify=expected,thread-local -pedantic -x c++ %s -DC11 -D__thread=_Thread_local -std=c++11 -Wno-deprecated
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -Wno-private-extern -verify=expected,thread-local -pedantic %s -std=c99 -D__thread=_Thread_local -DC99

#ifdef __cplusplus
// In C++, we define __private_extern__ to extern.
#undef __private_extern__
#endif

__thread int t1; // thread-local-warning {{'_Thread_local' is a C11 extension}}
__thread extern int t2; // thread-local-warning {{'_Thread_local' is a C11 extension}}
__thread static int t3; // thread-local-warning {{'_Thread_local' is a C11 extension}}
#ifdef GNU
// expected-warning@-3 {{'__thread' before 'extern'}}
// expected-warning@-3 {{'__thread' before 'static'}}
#endif

__thread __private_extern__ int t4; // thread-local-warning {{'_Thread_local' is a C11 extension}}
struct t5 { __thread int x; }; // thread-local-warning {{'_Thread_local' is a C11 extension}}
#ifdef __cplusplus
// expected-error-re@-2 {{'{{__thread|_Thread_local|thread_local}}' is only allowed on variable declarations}}
#else
// FIXME: The 'is only allowed on variable declarations' diagnostic is better here.
// expected-error@-5 {{type name does not allow storage class to be specified}}
#endif

__thread int t6(); // thread-local-warning {{'_Thread_local' is a C11 extension}}
#if defined(GNU)
// expected-error@-2 {{'__thread' is only allowed on variable declarations}}
#elif defined(C11) || defined(C99)
// expected-error@-4 {{'_Thread_local' is only allowed on variable declarations}}
#else
// expected-error@-6 {{'thread_local' is only allowed on variable declarations}}
#endif

int f(__thread int t7) { // expected-error {{' is only allowed on variable declarations}} \
                         // thread-local-warning {{'_Thread_local' is a C11 extension}}
  __thread int t8; // thread-local-warning {{'_Thread_local' is a C11 extension}}
#if defined(GNU)
  // expected-error@-2 {{'__thread' variables must have global storage}}
#elif defined(C11) || defined(C99)
  // expected-error@-4 {{'_Thread_local' variables must have global storage}}
#endif
  extern __thread int t9; // thread-local-warning {{'_Thread_local' is a C11 extension}}
  static __thread int t10; // thread-local-warning {{'_Thread_local' is a C11 extension}}
  __thread __private_extern__ int t11; // thread-local-warning {{'_Thread_local' is a C11 extension}}
#if __cplusplus < 201103L
  __thread auto int t12a; // expected-error-re {{cannot combine with previous '{{__thread|_Thread_local}}' declaration specifier}} \
                          // thread-local-warning {{'_Thread_local' is a C11 extension}}
  auto __thread int t12b; // expected-error {{cannot combine with previous 'auto' declaration specifier}} \
                          // thread-local-warning {{'_Thread_local' is a C11 extension}}
#elif !defined(CXX11)
  __thread auto t12a = 0; // expected-error {{'_Thread_local' variables must have global storage}} \
                          // thread-local-warning {{'_Thread_local' is a C11 extension}}
  auto __thread t12b = 0; // expected-error {{'_Thread_local' variables must have global storage}} \
                          // thread-local-warning {{'_Thread_local' is a C11 extension}}
#endif
  __thread register int t13a; // expected-error-re {{cannot combine with previous '{{__thread|_Thread_local|thread_local}}' declaration specifier}} \
                              // thread-local-warning {{'_Thread_local' is a C11 extension}}
  register __thread int t13b; // expected-error {{cannot combine with previous 'register' declaration specifier}} \
                              // thread-local-warning {{'_Thread_local' is a C11 extension}}
}

__thread typedef int t14; // expected-error-re {{cannot combine with previous '{{__thread|_Thread_local|thread_local}}' declaration specifier}} \
                          // thread-local-warning {{'_Thread_local' is a C11 extension}}
__thread int t15; // expected-note {{previous definition is here}} \
                  // thread-local-warning {{'_Thread_local' is a C11 extension}}
extern int t15; // expected-error {{non-thread-local declaration of 't15' follows thread-local declaration}}
extern int t16; // expected-note {{previous declaration is here}}
__thread int t16; // expected-error {{thread-local declaration of 't16' follows non-thread-local declaration}} \
                  // thread-local-warning {{'_Thread_local' is a C11 extension}}

#ifdef CXX11
extern thread_local int t17; // expected-note {{previous declaration is here}}
_Thread_local int t17; // expected-error {{thread-local declaration of 't17' with static initialization follows declaration with dynamic initialization}} \
                       // expected-warning {{'_Thread_local' is a C11 extension}}
extern _Thread_local int t18; // expected-note {{previous declaration is here}} \
                              // expected-warning {{'_Thread_local' is a C11 extension}}
thread_local int t18; // expected-error {{thread-local declaration of 't18' with dynamic initialization follows declaration with static initialization}}
#endif

// PR13720
__thread int thread_int; // thread-local-warning {{'_Thread_local' is a C11 extension}}
int *thread_int_ptr = &thread_int;
#ifndef __cplusplus
// expected-error@-2 {{initializer element is not a compile-time constant}}
#endif
void g() {
  int *p = &thread_int; // This is perfectly fine, though.
}
#if __cplusplus >= 201103L
constexpr int *thread_int_ptr_2 = &thread_int; // expected-error {{must be initialized by a constant expression}}
#endif

int non_const();
__thread int non_const_init = non_const(); // thread-local-warning {{'_Thread_local' is a C11 extension}}
#if !defined(__cplusplus)
// expected-error@-2 {{initializer element is not a compile-time constant}}
#elif !defined(CXX11)
// expected-error@-4 {{initializer for thread-local variable must be a constant expression}}
#if __cplusplus >= 201103L
// expected-note@-6 {{use 'thread_local' to allow this}}
#endif
#endif

#ifdef __cplusplus
struct S {
  ~S();
};
__thread S s; // thread-local-warning {{'_Thread_local' is a C11 extension}}
#if !defined(CXX11)
// expected-error@-2 {{type of thread-local variable has non-trivial destruction}}
#if __cplusplus >= 201103L
// expected-note@-4 {{use 'thread_local' to allow this}}
#endif
#endif
#endif

#ifdef __cplusplus
struct HasCtor {
  HasCtor();
};
__thread HasCtor var_with_ctor; // thread-local-warning {{'_Thread_local' is a C11 extension}}
#if !defined(CXX11)
// expected-error@-2 {{initializer for thread-local variable must be a constant expression}}
#if __cplusplus >= 201103L
// expected-note@-4 {{use 'thread_local' to allow this}}
#endif
#endif
#endif

__thread int aggregate[10] = {0}; // thread-local-warning {{'_Thread_local' is a C11 extension}}
