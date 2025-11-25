// UNSUPPORTED: system-windows
// REQUIRES: systemz-registered-target

// RUN: export CLANG_CONFIG_PATH=%S/Inputs/config-zos
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %clang --target=s390x-ibm-zos -c -### %s 2>&1 | FileCheck %s 
// CHECK: Configuration file: {{.*}}/Inputs/config-zos/clang.cfg
// CHECK: "-D" "ABC=123"

// RUN: export CLANG_CONFIG_PATH=%S/Inputs/config-zos/def.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %clang --target=s390x-ibm-zos -c -### %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEF
// CHECK-DEF: Configuration file: {{.*}}/Inputs/config-zos/def.cfg
// CHECK-DEF: "-D" "DEF=456"

// RUN: export CLANG_CONFIG_PATH=%S/Inputs/config-zos/Garbage
// RUN: env -u CLANG_NO_DEFAULT_CONFIG not %clang --target=s390x-ibm-zos -c -### %s 2>&1 | FileCheck %s  -check-prefix=CHECK-ERR
// CHECK-ERR:  error: configuration file '{{.*}}/Inputs/config-zos/Garbage' cannot be found

// The directory exists but no clang.cfg in it
// RUN: export CLANG_CONFIG_PATH=%S/Inputs/config-zos/tst
// RUN: env -u CLANG_NO_DEFAULT_CONFIG not %clang --target=s390x-ibm-zos -c -### %s 2>&1 | FileCheck %s  -check-prefix=CHECK-ERRDIR
// CHECK-ERRDIR:  error: configuration file '{{.*}}/Inputs/config-zos/tst/clang.cfg' cannot be found
