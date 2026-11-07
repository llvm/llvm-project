// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl -Xcc -I%S/Inputs 2>&1 | FileCheck %s \
// RUN:     --implicit-check-not="<<< inputs >>>"
// RUN: not %clang_cc1 -fincremental-extensions -fsyntax-only -I%S/Inputs \
// RUN:     %S/Inputs/include-stack-main.cpp 2>&1 | FileCheck %s --check-prefix=MAIN

// clang-repl includes each input from its main file. That is an
// implementation detail, which include stacks don't show.

// A main file that includes a header still shows in its include stack.
// MAIN: In file included from {{.*}}include-stack-main.cpp:1:
// MAIN-NEXT: {{.*}}include-stack.h:1:17: error: use of undeclared identifier 'undeclared_in_header'

int x = undeclared_in_input;
// CHECK: input_line_{{[0-9]+}}:1:9: error: use of undeclared identifier 'undeclared_in_input'

#include "include-stack.h"
// CHECK: In file included from input_line_{{[0-9]+}}:1:
// CHECK-NEXT: {{.*}}include-stack.h:1:17: error: use of undeclared identifier 'undeclared_in_header'

int y = undeclared_again;
// CHECK: input_line_{{[0-9]+}}:1:9: error: use of undeclared identifier 'undeclared_again'

%quit
