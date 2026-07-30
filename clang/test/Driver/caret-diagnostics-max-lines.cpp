//RUN: %clang -### -fcaret-diagnostics-max-lines=2 %s 2>&1 | FileCheck %s

// CHECK: "-fcaret-diagnostics-max-lines=2"
