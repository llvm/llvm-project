// RUN: %clang_pgogen  -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show -function main -counts  %t.profraw| FileCheck  %s 

void exit(int);
int g;
__attribute__((noinline)) void foo()
{
  g++;
  if (g==1000)
    exit(0);
}


int main()
{
  while (1) {
    foo();
  }

}

// CHECK: Counters:
// CHECK-NEXT:  main:
// CHECK-NEXT:    Hash: {{.*}}
// CHECK-NEXT:    Counters: 2
// CHECK-NEXT:    Block counts: [1000, 1]



