// RUN: %clangxx -O0 %s -o %t

// Recursive include: options1 includes options2
// RUN: echo "symbolize=1" > %t.options1.txt
// RUN: echo "include='%t.options2.txt'" >>%t.options1.txt
// RUN: echo "help=1" >%t.options2.txt
// RUN: echo "help=1" >%t.options.options-include.cc.tmp
// RUN: cat %t.options1.txt
// RUN: cat %t.options2.txt

// RUN: %env_tool_opts=help=0:include='"%t.options1.txt"' %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-WITH-HELP --check-prefix=CHECK-FOUND <%t.out

// RUN: %env_tool_opts=include='"%t.options1.txt"',help=0 %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-WITHOUT-HELP --check-prefix=CHECK-FOUND <%t.out

// RUN: %env_tool_opts=include='"%t.options-not-found.txt"',help=1 not %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-NOT-FOUND < %t.out

// include_if_exists does not fail when the file is missing
// RUN: %env_tool_opts=include_if_exists='"%t.options-not-found.txt"',help=1 %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-WITH-HELP --check-prefix=CHECK-FOUND < %t.out

// %b (binary basename substitution)
// RUN: %env_tool_opts=include='"%t.options.%b"' %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-WITH-HELP --check-prefix=CHECK-FOUND < %t.out

// RUN: %env_tool_opts=include='"%t.options-not-found.%b"' not %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-WITHOUT-HELP --check-prefix=CHECK-NOT-FOUND < %t.out

// RUN: %env_tool_opts=include_if_exists='"%t.options.%b"' %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-WITH-HELP --check-prefix=CHECK-FOUND < %t.out

// RUN: %env_tool_opts=include_if_exists='"%t.options-not-found.%b"' %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-WITHOUT-HELP --check-prefix=CHECK-FOUND < %t.out


#include <stdio.h>

int main() {
  fprintf(stderr, "done\n");
}

// CHECK-WITH-HELP: Available flags for
// CHECK-WITHOUT-HELP-NOT: Available flags for
// CHECK-FOUND-NOT: Failed to read options from
// CHECK-NOT-FOUND: Failed to read options from
