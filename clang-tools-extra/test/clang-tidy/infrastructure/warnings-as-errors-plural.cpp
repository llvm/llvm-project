// RUN: clang-tidy %s -checks='-*,llvm-namespace-comment,clang-diagnostic*' -- 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WARN -implicit-check-not='{{warning|error}}:'
// RUN: not clang-tidy %s -checks='-*,llvm-namespace-comment,clang-diagnostic*' \
// RUN:   -warnings-as-errors='llvm-namespace-comment' -- 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WERR -implicit-check-not='{{warning|error}}:'
// RUN: not clang-tidy %s -checks='-*,llvm-namespace-comment,clang-diagnostic*' \
// RUN:   -warnings-as-errors='llvm-namespace-comment' -quiet -- 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WERR-QUIET -implicit-check-not='{{warning|error}}:'

namespace j {
}
// CHECK-WARN: warning: namespace 'j' not terminated with a closing comment [llvm-namespace-comment]
// CHECK-WERR: error: namespace 'j' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]
// CHECK-WERR-QUIET: error: namespace 'j' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]

namespace k {
}
// CHECK-WARN: warning: namespace 'k' not terminated with a closing comment [llvm-namespace-comment]
// CHECK-WERR: error: namespace 'k' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]
// CHECK-WERR-QUIET: error: namespace 'k' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]

// CHECK-WARN-NOT: treated as
// CHECK-WERR: 2 warnings treated as errors
// CHECK-WERR-QUIET-NOT: treated as
