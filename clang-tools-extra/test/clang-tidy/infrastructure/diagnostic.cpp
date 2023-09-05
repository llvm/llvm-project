// RUN: not clang-tidy -checks='-*,modernize-use-override' %s.nonexistent.cpp -- | FileCheck -check-prefix=CHECK1 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %s -- -fan-unknown-option | FileCheck -check-prefix=CHECK2 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,google-explicit-constructor,clang-diagnostic-literal-conversion' %s -- -fan-unknown-option | FileCheck -check-prefix=CHECK3 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-macro-redefined' %s -- -DMACRO_FROM_COMMAND_LINE | FileCheck -check-prefix=CHECK4 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,modernize-use-override' %s -- -DCOMPILATION_ERROR | FileCheck -check-prefix=CHECK6 -implicit-check-not='{{warning:|error:}}' %s
//
// Now repeat the tests and ensure no other errors appear on stderr:
// RUN: not clang-tidy -checks='-*,modernize-use-override' %s.nonexistent.cpp -- 2>&1 | FileCheck -check-prefix=CHECK1 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %s -- -fan-unknown-option 2>&1 | FileCheck -check-prefix=CHECK2 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,google-explicit-constructor,clang-diagnostic-literal-conversion' %s -- -fan-unknown-option 2>&1 | FileCheck -check-prefix=CHECK3 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-macro-redefined' %s -- -DMACRO_FROM_COMMAND_LINE 2>&1 | FileCheck -check-prefix=CHECK4 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,modernize-use-override' %s -- -DCOMPILATION_ERROR 2>&1 | FileCheck -check-prefix=CHECK6 -implicit-check-not='{{warning:|error:}}' %s
//
// Now create a directory with a compilation database file and ensure we don't
// use it after failing to parse commands from the command line:
//
// RUN: mkdir -p %T/diagnostics/
// RUN: echo '[{"directory": "%/T/diagnostics/","command": "clang++ -fan-option-from-compilation-database -c %/T/diagnostics/input.cpp", "file": "%/T/diagnostics/input.cpp"}]' > %T/diagnostics/compile_commands.json
// RUN: cat %s > %T/diagnostics/input.cpp
// RUN: not clang-tidy -checks='-*,modernize-use-override' %T/diagnostics/nonexistent.cpp -- 2>&1 | FileCheck -check-prefix=CHECK1 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %T/diagnostics/input.cpp -- -fan-unknown-option 2>&1 | FileCheck -check-prefix=CHECK2 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,google-explicit-constructor,clang-diagnostic-literal-conversion' %T/diagnostics/input.cpp -- -fan-unknown-option 2>&1 | FileCheck -check-prefix=CHECK3 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-macro-redefined' %T/diagnostics/input.cpp -- -DMACRO_FROM_COMMAND_LINE 2>&1 | FileCheck -check-prefix=CHECK4 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %T/diagnostics/input.cpp 2>&1 | FileCheck -check-prefix=CHECK5 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,modernize-use-override' %T/diagnostics/input.cpp -- -DCOMPILATION_ERROR 2>&1 | FileCheck -check-prefix=CHECK6 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-macro-redefined' %s -- -DMACRO_FROM_COMMAND_LINE -std=c++20 | FileCheck -check-prefix=CHECK4 -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-macro-redefined,clang-diagnostic-literal-conversion' %s -- -DMACRO_FROM_COMMAND_LINE -std=c++20 -Wno-macro-redefined | FileCheck --check-prefix=CHECK7 -implicit-check-not='{{warning:|error:}}' %s
// RUN: not clang-tidy -checks='-*,modernize-use-override' %s -- -std=c++20 -DPR64602 | FileCheck -check-prefix=CHECK8 -implicit-check-not='{{warning:|error:}}' %s

// CHECK1: error: no input files [clang-diagnostic-error]
// CHECK1: error: no such file or directory: '{{.*}}nonexistent.cpp' [clang-diagnostic-error]
// CHECK1: error: unable to handle compilation{{.*}} [clang-diagnostic-error]
// CHECK2: error: unknown argument: '-fan-unknown-option' [clang-diagnostic-error]
// CHECK3: error: unknown argument: '-fan-unknown-option' [clang-diagnostic-error]
// CHECK5: error: unknown argument: '-fan-option-from-compilation-database' [clang-diagnostic-error]

// CHECK7: :[[@LINE+4]]:9: warning: implicit conversion from 'double' to 'int' changes value from 1.5 to 1 [clang-diagnostic-literal-conversion]
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
// CHECK7-NOT: :[[@LINE-2]]:9: warning: 'MACRO_FROM_COMMAND_LINE' macro redefined

#ifdef COMPILATION_ERROR
void f(int a) {
  &(a + 1);
  // CHECK6: :[[@LINE-1]]:3: error: cannot take the address of an rvalue of type 'int' [clang-diagnostic-error]
}
#endif

#ifdef PR64602 // Should not crash
template <class T = void>
struct S
{
    auto foo(auto);
};

template <>
auto S<>::foo(auto)
{
    return 1;
}
// CHECK8: error: template parameter list matching the non-templated nested type 'S<>' should be empty ('template<>') [clang-diagnostic-error]
#endif
