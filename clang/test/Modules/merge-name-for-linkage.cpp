// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/merge-name-for-linkage -verify %s
// expected-no-diagnostics
typedef union {} pthread_mutex_t;
typedef pthread_mutex_t pthread_mutex_t;
#include "a.h"
pthread_mutex_t x;
#include "b.h"
pthread_mutex_t y;
merged_after_definition z;
