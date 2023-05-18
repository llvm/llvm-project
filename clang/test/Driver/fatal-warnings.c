// RUN: %clang -### %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -Wa,--fatal-warnings 2>&1 | FileCheck %s
// RUN: not %clang %s -c -o %t.o -target i686-pc-linux-gnu -integrated-as -Wa,--fatal-warnings 2>&1 %t.log
// FileCheck --check-prefix=CHECK-AS %s -input-file %t.log

// CHECK: "-cc1" {{.*}} "-massembler-fatal-warnings"
// CHECK-AS: error: .warning argument must be a string

__asm(".warning 1");
