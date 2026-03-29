// RUN: clang-tidy --checks="-*,modernize-loop-convert" \
// RUN:   --config="CheckOptions: { \
// RUN:      modernize-loop-convert.NamingStyle: '0' }" \
// RUN:   --dump-config -- | FileCheck %s

// CHECK: modernize-loop-convert.NamingStyle: CamelCase
