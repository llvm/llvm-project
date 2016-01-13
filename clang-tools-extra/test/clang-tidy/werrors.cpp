// RUN: clang-tidy %s -checks='-*,llvm-namespace-comment' -- 2>&1 | FileCheck %s --check-prefix=CHECK-WARN
// RUN: not clang-tidy %s -checks='-*,llvm-namespace-comment' -warnings-as-errors='llvm-namespace-comment' -- 2>&1 | FileCheck %s --check-prefix=CHECK-WERR

namespace i {
}
// CHECK-WARN: warning: namespace 'i' not terminated with a closing comment [llvm-namespace-comment]
// CHECK-WERR: error: namespace 'i' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]

// CHECK-WARN-NOT: treated as
// CHECK-WERR: 1 warning treated as error
