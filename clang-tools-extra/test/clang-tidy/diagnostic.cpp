// RUN: clang-tidy -checks='-*,modernize-use-override' %s.nonexistent.cpp -- | FileCheck -check-prefix=CHECK1 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %s -- -fan-unknown-option | FileCheck -check-prefix=CHECK2 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor,clang-diagnostic-literal-conversion' %s -- -fan-unknown-option | FileCheck -check-prefix=CHECK3 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-macro-redefined' %s -- -DMACRO_FROM_COMMAND_LINE | FileCheck -check-prefix=CHECK4 -implicit-check-not='{{warning:|error:}}' %s
//
// Now repeat the tests and ensure no other errors appear on stderr:
// RUN: clang-tidy -checks='-*,modernize-use-override' %s.nonexistent.cpp -- 2>&1 | FileCheck -check-prefix=CHECK1 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %s -- -fan-unknown-option 2>&1 | FileCheck -check-prefix=CHECK2 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor,clang-diagnostic-literal-conversion' %s -- -fan-unknown-option 2>&1 | FileCheck -check-prefix=CHECK3 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-macro-redefined' %s -- -DMACRO_FROM_COMMAND_LINE 2>&1 | FileCheck -check-prefix=CHECK4 -implicit-check-not='{{warning:|error:}}' %s
//
// Now create a directory with a compilation database file and ensure we don't
// use it after failing to parse commands from the command line:
//
// RUN: mkdir -p %T/diagnostics/
// RUN: echo '[{"directory": "%/T/diagnostics/","command": "clang++ -fan-option-from-compilation-database -c %/T/diagnostics/input.cpp", "file": "%/T/diagnostics/input.cpp"}]' > %T/diagnostics/compile_commands.json
// RUN: cat %s > %T/diagnostics/input.cpp
// RUN: clang-tidy -checks='-*,modernize-use-override' %T/diagnostics/nonexistent.cpp -- 2>&1 | FileCheck -check-prefix=CHECK1 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %T/diagnostics/input.cpp -- -fan-unknown-option 2>&1 | FileCheck -check-prefix=CHECK2 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor,clang-diagnostic-literal-conversion' %T/diagnostics/input.cpp -- -fan-unknown-option 2>&1 | FileCheck -check-prefix=CHECK3 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-macro-redefined' %T/diagnostics/input.cpp -- -DMACRO_FROM_COMMAND_LINE 2>&1 | FileCheck -check-prefix=CHECK4 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %T/diagnostics/input.cpp 2>&1 | FileCheck -check-prefix=CHECK5 -implicit-check-not='{{warning:|error:}}' %s

// CHECK1: error: error reading '{{.*}}nonexistent.cpp' [clang-diagnostic-error]
// CHECK2: error: unknown argument: '-fan-unknown-option' [clang-diagnostic-error]
// CHECK3: error: unknown argument: '-fan-unknown-option' [clang-diagnostic-error]
// CHECK5: error: unknown argument: '-fan-option-from-compilation-database' [clang-diagnostic-error]

// CHECK2: :[[@LINE+3]]:9: warning: implicit conversion from 'double' to 'int' changes value from 1.5 to 1 [clang-diagnostic-literal-conversion]
// CHECK3: :[[@LINE+2]]:9: warning: implicit conversion from 'double' to 'int' changes value
// CHECK5: :[[@LINE+1]]:9: warning: implicit conversion from 'double' to 'int' changes value
int a = 1.5;

// CHECK2: :[[@LINE+3]]:11: warning: single-argument constructors must be marked explicit
// CHECK3: :[[@LINE+2]]:11: warning: single-argument constructors must be marked explicit
// CHECK5: :[[@LINE+1]]:11: warning: single-argument constructors must be marked explicit
class A { A(int) {} };

#define MACRO_FROM_COMMAND_LINE
// CHECK4: :[[@LINE-1]]:9: warning: 'MACRO_FROM_COMMAND_LINE' macro redefined
