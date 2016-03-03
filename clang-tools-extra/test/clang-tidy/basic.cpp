// RUN: clang-tidy %s -checks='-*,llvm-namespace-comment' -- | FileCheck %s
// RUN: c-index-test -test-load-source-reparse 2 all %s -Xclang -add-plugin -Xclang clang-tidy -Xclang -plugin-arg-clang-tidy -Xclang -checks='-*,llvm-namespace-comment' 2>&1 | FileCheck %s

namespace i {
}
// CHECK: warning: namespace 'i' not terminated with a closing comment [llvm-namespace-comment]
