// RUN: %check_clang_tidy %s bugprone-macro-condition %t

#define USE_FOO 0
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: Macro 'USE_FOO' defined here with a value and checked for definition

#if defined(USE_FOO)
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: Macro 'USE_FOO' defined with a value and checked here for definition
void f()
{
  extern void foo();
  foo();
}
#endif

#if 0
#elif OTHER_MACRO
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Undefined macro 'OTHER_MACRO' checked here for value
#elifdef OTHER_MACRO2
#else
#endif

#if !defined(USE_FOO)
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: Macro 'USE_FOO' defined with a value and checked here for definition
void f2()
{
  extern void notFoo();
  notFoo();
}
#endif

#ifdef USE_FOO
// CHECK-MESSAGES: :[[@LINE-1]]:2: warning: Macro 'USE_FOO' defined with a value and checked here for definition
void f3()
{
  extern void foo();
  foo();
}
#endif

#ifndef USE_FOO
// CHECK-MESSAGES: :[[@LINE-1]]:2: warning: Macro 'USE_FOO' defined with a value and checked here for definition
void f4()
{
  extern void notFoo();
  notFoo();
}
#endif

#if 0
#elif defined(USE_FOO)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Macro 'USE_FOO' defined with a value and checked here for definition
void f5()
{
  extern void foo();
  foo();
}
#endif

#define USE_GRONK 0
#ifdef USE_GRONK
#if USE_GRONK
void f6()
{
  extern void foo();
  foo();
}
#endif
#endif

#if 0
#elif defined(USE_GRONK)
#if USE_GRONK
void f7()
{
  extern void foo();
  foo();
}
#endif
#endif

#if defined(USE_GRONK) && USE_GRONK
void f8()
{
  extern void foo();
  foo();
}
#endif
