// REQUIRES: x86-registered-target
// RUN: not %clang -target x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=bar-format %s 2>&1 | FileCheck %s

// RUN: not %clang -target x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-tapi-elf-v1 %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-TAPI-DEPRECATED %s

// RUN: not %clang -target x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-yaml-elf-v1 %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-YAML-DEPRECATED %s

// RUN: not %clang_cc1 -target x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=bar-format %s 2>&1 | FileCheck %s

// RUN: not %clang_cc1 -target x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-tapi-elf-v1 %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-TAPI-DEPRECATED %s

// RUN: not %clang_cc1 -target x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-yaml-elf-v1 %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-YAML-DEPRECATED %s

// CHECK: error: invalid value
// CHECK: 'Invalid interface stub format: bar-format.' in 'Must specify a
// CHECK: valid interface stub format type, ie:
// CHECK: -interface-stub-version=experimental-ifs-v1'

// CHECK-TAPI-DEPRECATED: error: invalid value
// CHECK-TAPI-DEPRECATED: 'Invalid interface stub format:
// CHECK-TAPI-DEPRECATED: experimental-tapi-elf-v1 is deprecated.' in 'Must
// CHECK-TAPI-DEPRECATED: specify a valid interface stub format type, ie:
// CHECK-TAPI-DEPRECATED: -interface-stub-version=experimental-ifs-v1'

// CHECK-YAML-DEPRECATED: error: invalid value
// CHECK-YAML-DEPRECATED: 'Invalid interface stub format:
// CHECK-YAML-DEPRECATED: experimental-yaml-elf-v1 is deprecated.' in 'Must
// CHECK-YAML-DEPRECATED: specify a valid interface stub format type, ie:
// CHECK-YAML-DEPRECATED: -interface-stub-version=experimental-ifs-v1'
