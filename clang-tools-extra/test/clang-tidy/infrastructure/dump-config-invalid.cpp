// RUN: clang-tidy --checks="-*,modernize-make-shared" \
// RUN:   --config="CheckOptions: { \
// RUN:      modernize-make-shared.IncludeStyle: '0' }" \
// RUN:   --dump-config -- | FileCheck %s

// CHECK: modernize-make-shared.IncludeStyle: llvm
