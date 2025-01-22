// REQUIRES: shell
// REQUIRES: systemz-registered-target

// RUN: unset CLANG_NO_DEFAULT_CONFIG

// RUN: export CLANG_CONFIG_PATH=%S/Inputs/config-zos
// RUN: %clang --target=s390x-ibm-zos -c -### %s 2>&1 | FileCheck %s 
// CHECK: Configuration file: {{.*}}/Inputs/config-zos/clang.cfg
// CHECK: "-D" "ABC=123"

// RUN: export CLANG_CONFIG_PATH=%S/Inputs/config-zos/def.cfg
// RUN: %clang --target=s390x-ibm-zos -c -### %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEF
// CHECK-DEF: Configuration file: {{.*}}/Inputs/config-zos/def.cfg
// CHECK-DEF: "-D" "DEF=456"

// RUN: export CLANG_CONFIG_PATH=%S/Inputs/config-zos/Garbage
// RUN: not %clang --target=s390x-ibm-zos -c -### %s 2>&1 | FileCheck %s  -check-prefix=CHECK-ERR
// CHECK-ERR:  error: configuration file '{{.*}}/Inputs/config-zos/Garbage' cannot be found

// The directory exists but no clang.cfg in it
// RUN: export CLANG_CONFIG_PATH=%S/Inputs/config-zos/tst
// RUN: not %clang --target=s390x-ibm-zos -c -### %s 2>&1 | FileCheck %s  -check-prefix=CHECK-ERRDIR
// CHECK-ERRDIR:  error: configuration file '{{.*}}/Inputs/config-zos/tst/clang.cfg' cannot be found
