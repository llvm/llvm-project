// https://github.com/llvm/llvm-project/issues/163369
// RUN: %clang_tsan %s -o %t && %run %t

#if __x86_64__
__attribute__((target_clones("avx,default")))
#endif
static int has_target_clones(void) {
  return 0;
}

int main(void) { has_target_clones(); }
