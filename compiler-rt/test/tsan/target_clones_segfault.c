// https://github.com/llvm/llvm-project/issues/163369
// RUN: %clang_tsan %s -o %t && %run %t

__attribute__((target_clones("default,avx"))) static int
has_target_clones(void) {
  return 0;
}

int main(void) { has_target_clones(); }
