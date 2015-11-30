// RUN: clang-tidy -checks='-*,modernize-use-override' %s.nonexistent.cpp -- | FileCheck -check-prefix=CHECK1 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %s -- -fan-unknown-option | FileCheck -check-prefix=CHECK2 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor,clang-diagnostic-literal-conversion' %s -- -fan-unknown-option | FileCheck -check-prefix=CHECK3 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-macro-redefined' %s -- -DMACRO_FROM_COMMAND_LINE | FileCheck -check-prefix=CHECK4 -implicit-check-not='{{warning:|error:}}' %s

// CHECK1: error: error reading '{{.*}}.nonexistent.cpp' [clang-diagnostic-error]
// CHECK2: error: unknown argument: '-fan-unknown-option' [clang-diagnostic-error]
// CHECK3: error: unknown argument: '-fan-unknown-option' [clang-diagnostic-error]

// CHECK2: :[[@LINE+2]]:9: warning: implicit conversion from 'double' to 'int' changes value from 1.5 to 1 [clang-diagnostic-literal-conversion]
// CHECK3: :[[@LINE+1]]:9: warning: implicit conversion from 'double' to 'int' changes value
int a = 1.5;

// CHECK2: :[[@LINE+2]]:11: warning: single-argument constructors must be marked explicit
// CHECK3: :[[@LINE+1]]:11: warning: single-argument constructors must be marked explicit
class A { A(int) {} };

#define MACRO_FROM_COMMAND_LINE
// CHECK4: :[[@LINE-1]]:9: warning: 'MACRO_FROM_COMMAND_LINE' macro redefined
