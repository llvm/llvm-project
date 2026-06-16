// RUN: not clang-tidy --vfsoverlay=%t.nonexistent.yaml \
// RUN:     -checks='misc-explicit-constructor' %s -- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-MISSING
// CHECK-MISSING: Can't load virtual filesystem overlay file '{{.*}}.nonexistent.yaml':

// Malformed overlay YAML.
// RUN: echo 'not valid yaml: ][' > %t.invalid.yaml
// RUN: not clang-tidy --vfsoverlay=%t.invalid.yaml \
// RUN:     -checks='misc-explicit-constructor' %s -- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-INVALID
// CHECK-INVALID: Error: invalid virtual filesystem overlay file '{{.*}}.invalid.yaml'.
