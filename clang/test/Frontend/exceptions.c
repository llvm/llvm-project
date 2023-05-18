// RUN: %clang_cc1 -fms-compatibility -fexceptions -fcxx-exceptions -DMS_MODE -verify %s
// expected-no-diagnostics

// RUN: %clang_cc1 -fms-compatibility -fexceptions -verify %s
// expected-no-diagnostics

#if defined(MS_MODE) && defined(__EXCEPTIONS)
#error __EXCEPTIONS should not be defined.
#endif
