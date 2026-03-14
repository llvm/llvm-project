// Regression test for https://bugs.llvm.org/show_bug.cgi?id=32842
//
// RUN: %clang_msan -g %s -o %t
// RUN: not %run %t 2>&1 | FileCheck  %s

struct iphdr {
  unsigned char pad1: 2, ihl:4, pad2: 2;
};

int raw_send_hdrinc(unsigned long int length) {
  struct iphdr iph;
  if (iph.ihl * 4 > length) {
    return 1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  return raw_send_hdrinc(12);
}

// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
