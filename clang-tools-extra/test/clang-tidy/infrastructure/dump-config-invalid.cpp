// RUN: clang-tidy --checks="-*,modernize-make-shared" \
// RUN:   --config="CheckOptions: [{ \
// RUN:      key: modernize-make-shared.IncludeStyle, value: '0' }]" \
// RUN:   --dump-config -- | FileCheck %s

// CHECK: modernize-make-shared.IncludeStyle: llvm
