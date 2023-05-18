// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -verify %s
// expected-no-diagnostics

#pragma clang module build sys_types
module sys_types {}
#pragma clang module contents
#pragma clang module begin sys_types
extern "C" {
  typedef union { bool b; } pthread_mutex_t;
}
#pragma clang module end
#pragma clang module endbuild

typedef union { bool b; } pthread_mutex_t;
#pragma clang module import sys_types

const pthread_mutex_t *m;

