// RUN: %clangxx_msan %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=MISSED --allow-empty < %t.out
// RUN: %clangxx_msan %s -mllvm -msan-eager-checks=1 -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan %s -disable-noundef-analysis -fsanitize-memory-param-retval -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

struct SimpleStruct {
  int md1;
};

static int sink;

static void examineValue(int x) { sink = x; }

int main(int argc, char *argv[]) {
  auto ss = new SimpleStruct;
  examineValue(ss->md1);

  return 0;
}

// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
// MISSED-NOT: use-of-uninitialized-value
