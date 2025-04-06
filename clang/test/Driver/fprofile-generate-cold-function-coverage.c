// RUN: %clang -### -c -fprofile-generate-cold-function-coverage %s 2>&1 | FileCheck %s
// CHECK: "--instrument-cold-function-only-path=default_%m.profraw" 
// CHECK: "--pgo-instrument-cold-function-only"
// CHECK: "--pgo-function-entry-coverage"
// CHECK-NOT:  "-fprofile-instrument"
// CHECK-NOT:  "-fprofile-instrument-path=

// RUN: %clang -### -c -fprofile-generate-cold-function-coverage=dir %s 2>&1 | FileCheck %s --check-prefix=CHECK-EQ
// CHECK-EQ: "--instrument-cold-function-only-path=dir{{/|\\\\}}default_%m.profraw" 
