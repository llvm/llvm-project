// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: not %clang -target x86_64-unknown-unknown -extract-api %t/first-header.h -x objective-c-header %t/second-header.h 2>&1 | FileCheck %s

// CHECK: error: header file
// CHECK-SAME: input 'objective-c-header' does not match the type of prior input in api extraction; use '-x c-header' to override

//--- first-header.h

void dummy_function(void);

//--- second-header.h

void other_dummy_function(void);
