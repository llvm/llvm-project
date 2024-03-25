// RUN: %clangxx_pgogen %s -O2 -g -o %s.bin -fno-exceptions -mllvm -profile-context-root=the_root
// RUN: %s.bin %t.rawprof
// RUN: %llvm-ctx-ifdo %t.rawprof %t.bitstream
// RUN: stat -c%%s %t.rawprof | FileCheck %s --check-prefix=RAW
// RUN: llvm-bcanalyzer --dump %t.bitstream 2>&1 | FileCheck %s --check-prefix=BC

// RAW: 1048592
// BC:      <UnknownBlock100 NumWords=4 BlockCodeSize=2>
// BC-NEXT:   <UnknownCode1 op0=-7380956406374790822/>
// BC-NEXT:   <UnknownCode3 op0=1/>
// BC-NEXT: </UnknownBlock100>

#include <cstdio>
extern "C" int __llvm_ctx_profile_dump(const char *Filename);

extern "C" {
void someFunction() { printf("check 2\n"); }

// block inlining because the pre-inliner otherwise will inline this - it's
// too small.
__attribute__((noinline)) void the_root() {
  printf("check 1\n");
  someFunction();
  someFunction();
}
}

int main(int argc, char **argv) {
  the_root();
  return __llvm_ctx_profile_dump(argv[1]);
}