// RUN: clang-tidy %s -checks='-*,llvm-namespace-comment' -- 2>&1 | FileCheck %s --check-prefix=CHECK-WARN -implicit-check-not='{{warning|error}}:'
// RUN: not clang-tidy %s -checks='-*,llvm-namespace-comment' -warnings-as-errors='llvm-namespace-comment' -- 2>&1 | FileCheck %s --check-prefix=CHECK-WERR -implicit-check-not='{{warning|error}}:'
// RUN: not clang-tidy %s -checks='-*,llvm-namespace-comment' -warnings-as-errors='llvm-namespace-comment' -quiet -- 2>&1 | FileCheck %s --check-prefix=CHECK-WERR-QUIET -implicit-check-not='{{warning|error}}:'

namespace i {
}
// CHECK-WARN: warning: namespace 'i' not terminated with a closing comment [llvm-namespace-comment]
// CHECK-WERR: error: namespace 'i' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]
// CHECK-WERR-QUIET: error: namespace 'i' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]

// CHECK-WARN-NOT: treated as
// CHECK-WERR: 1 warning treated as error
// CHECK-WERR-QUIET-NOT: treated as
