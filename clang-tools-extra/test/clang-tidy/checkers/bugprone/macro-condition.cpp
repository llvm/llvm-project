// RUN: %check_clang_tidy %s bugprone-macro-condition %t

#define USE_FOO 0

#if defined(USE_FOO)
void f()
{
  extern void foo();
  foo();
}
#endif

#define VALUE_DEFINED 42
#ifndef VALUE_DEFINED
#endif
// CHECK-MESSAGES-NOT: warning: Macro 'VALUE_DEFINED'

#if 0
#elif OTHER_MACRO
#elifdef OTHER_MACRO2
#else
#endif

#if !defined(USE_FOO)
void f2()
{
  extern void notFoo();
  notFoo();
}
#endif

#ifdef USE_FOO
void f3()
{
  extern void foo();
  foo();
}
#endif

#ifndef USE_FOO
void f4()
{
  extern void notFoo();
  notFoo();
}
#endif

#if 0
#elif defined(USE_FOO)
void f5()
{
  extern void foo();
  foo();
}
#endif

// CHECK-MESSAGES-NOT: warning: Macro 'USE_FOO'
// CHECK-MESSAGES-NOT: warning: Undefined macro 'OTHER_MACRO'

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
// CHECK-MESSAGES-NOT: warning: Macro 'USE_GRONK'

#define ENCLOSED_DEFINED 1
#if defined(ENCLOSED_DEFINED)
#if ENCLOSED_DEFINED
#endif
#endif
// CHECK-MESSAGES-NOT: warning: Macro 'ENCLOSED_DEFINED'

#define ENCLOSED_CONJUNCTION 1
#define ENCLOSING_SUPPORT 1
#if defined(ENCLOSED_CONJUNCTION) && defined(ENCLOSING_SUPPORT)
#if ENCLOSED_CONJUNCTION
#endif
#endif
// CHECK-MESSAGES-NOT: warning: Macro 'ENCLOSED_CONJUNCTION'

#define VALIDATED_VALUE 100
#ifdef VALIDATED_VALUE
#endif
#if VALIDATED_VALUE < 10
#error VALIDATED_VALUE is too small.
#endif
// CHECK-MESSAGES-NOT: warning: Macro 'VALIDATED_VALUE'

#define CONDITIONAL_ERROR 1
#ifdef CONDITIONAL_ERROR
#endif
#if CONDITIONAL_ERROR
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: Macro 'CONDITIONAL_ERROR' checked here for value after being checked for definition
// CHECK-MESSAGES: :[[@LINE-4]]:2: note: Macro 'CONDITIONAL_ERROR' first checked here for definition
#if 0
#error This error is conditional.
#endif
#endif

#define VALUE_FIRST 0
#if VALUE_FIRST
#endif
#ifndef VALUE_FIRST
void f7()
{
  extern void foo();
  foo();
}
#endif
// CHECK-MESSAGES-NOT: warning: Macro 'VALUE_FIRST'

#define REQUIRED_OPTION 1
#define REQUIRED_NAME required_namespace
#if !defined(REQUIRED_OPTION) || \
    !defined(REQUIRED_NAME)
#error Required options are not configured.
#endif
#if defined(__cplusplus) && REQUIRED_OPTION == 1
#endif
// CHECK-MESSAGES-NOT: warning: Macro 'REQUIRED_OPTION'

#define POSITIVE_DEFINITION 1
#define OTHER_POSITIVE_DEFINITION 1
#if defined(POSITIVE_DEFINITION) || defined(OTHER_POSITIVE_DEFINITION)
#endif
#if POSITIVE_DEFINITION
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: Macro 'POSITIVE_DEFINITION' checked here for value after being checked for definition
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: Macro 'POSITIVE_DEFINITION' first checked here for definition
#endif

#define SAME_CONDITION 0
#if defined(SAME_CONDITION) && SAME_CONDITION
void f8()
{
  extern void foo();
  foo();
}
#endif

#if __has_include(<sys/file.h>)
#include <sys/file.h>
#endif
// CHECK-MESSAGES-NOT: warning: Undefined macro 'sys' checked here for value
// CHECK-MESSAGES-NOT: warning: Undefined macro 'file' checked here for value
// CHECK-MESSAGES-NOT: warning: Undefined macro 'h' checked here for value

#if __has_builtin(__builtin_trap)
#endif
// CHECK-MESSAGES-NOT: warning: Undefined macro '__builtin_trap' checked here for value

#if __has_cpp_attribute(gnu::always_inline)
#endif
// CHECK-MESSAGES-NOT: warning: Undefined macro 'gnu' checked here for value
// CHECK-MESSAGES-NOT: warning: Undefined macro 'always_inline' checked here for value

#define ALWAYS_TRUE(x) 1
#if ALWAYS_TRUE(not_a_macro)
#endif
// CHECK-MESSAGES-NOT: warning: Undefined macro 'not_a_macro' checked here for value

#ifdef ALWAYS_TRUE
#endif
// CHECK-MESSAGES-NOT: warning: Macro 'ALWAYS_TRUE' defined here with a value and checked for definition

#define GUARDED 1
#if defined(GUARDED)
#undef GUARDED
#if GUARDED
#endif
#endif
// CHECK-MESSAGES-NOT: warning: {{.*}}'GUARDED'

#define GUARDED_CONJUNCTION 1
#ifdef GUARDED_CONJUNCTION
#endif
#if defined(GUARDED_CONJUNCTION) && GUARDED_CONJUNCTION
#endif

#define GUARDED_DISJUNCTION 1
#ifdef GUARDED_DISJUNCTION
#endif
#if !defined(GUARDED_DISJUNCTION) || GUARDED_DISJUNCTION
#endif

#ifdef __cplusplus
#endif
#if __cplusplus >= 201103L
#endif

#ifdef __STDC_HOSTED__
#endif
#if __STDC_HOSTED__
#endif

#define DEFAULT_IFNDEF 1
#ifndef DEFAULT_IFNDEF
#define DEFAULT_IFNDEF 0
#endif
#if DEFAULT_IFNDEF
#endif

#define DEFAULT_NOT_DEFINED 1
#if !defined(DEFAULT_NOT_DEFINED)
#define DEFAULT_NOT_DEFINED 0
#endif
#if DEFAULT_NOT_DEFINED
#endif

#define STRINGIFY_IMPL(VALUE) #VALUE
#define STRINGIFY(VALUE) STRINGIFY_IMPL(VALUE)

#define REPORTED_DEFINITION_FIRST 10
#ifdef REPORTED_DEFINITION_FIRST
const char *reportedDefinitionFirst = STRINGIFY(REPORTED_DEFINITION_FIRST);
#endif
#if REPORTED_DEFINITION_FIRST > 0
#endif
// CHECK-MESSAGES-NOT: warning: Macro 'REPORTED_DEFINITION_FIRST'

#define REPORTED_VALUE_FIRST 10
#if REPORTED_VALUE_FIRST > 0
#endif
#ifdef REPORTED_VALUE_FIRST
const char *reportedValueFirst = STRINGIFY(REPORTED_VALUE_FIRST);
#endif
// CHECK-MESSAGES-NOT: warning: Macro 'REPORTED_VALUE_FIRST'
