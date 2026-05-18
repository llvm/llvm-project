// RUN: not clang-tidy %s @%S/Inputs/param/parameters.txt -- | FileCheck %s

namespace i {
}
// CHECK: error: namespace 'i' not terminated with a closing comment [llvm-namespace-comment,-warnings-as-errors]
