// RUN: clang-tidy %s -checks='-*,llvm-namespace-comment' -- 2>&1 | FileCheck %s --check-prefix=CHECK-WARN
// RUN: not clang-tidy %s -checks='-*,llvm-namespace-comment' -warnings-as-errors='llvm-namespace-comment' -- 2>&1 | FileCheck %s --check-prefix=CHECK-WERR

namespace j {
}
// CHECK-WARN: warning: namespace 'j' not terminated with a closing comment [llvm-namespace-comment]
// CHECK-WERR: error: namespace 'j' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]

namespace k {
}
// CHECK-WARN: warning: namespace 'k' not terminated with a closing comment [llvm-namespace-comment]
// CHECK-WERR: error: namespace 'k' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]

// CHECK-WARN-NOT: treated as
// CHECK-WERR: 2 warnings treated as errors
