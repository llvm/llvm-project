// Make sure that arguments that begin with @ are left as is in the argument
// stream, and also that @file arguments continue to be processed.

// RUN: echo "-D FOO" > %t.args
// RUN: %clang -rpath @executable_path/../lib @%t.args %s -### 2>&1 | FileCheck %s
// CHECK: "-D" "FOO"
// CHECK: "-rpath" "@executable_path/../lib"
