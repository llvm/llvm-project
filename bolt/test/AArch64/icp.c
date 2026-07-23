// This test verifies basic indirect call promotion on AArch64.

// RUN: %clang %cflags -O0 %s -o %t.exe -Wl,-q
// RUN: link_fdata %s %t.exe %t.fdata
// RUN: llvm-bolt %t.exe --icp=calls --icp-calls-topn=1 \
// RUN:   -o %t.null --lite=0 --assume-abi --print-icp --data %t.fdata \
// RUN:   | FileCheck %s

// CHECK: Binary Function "execute" after indirect-call-promotion
// CHECK: adrp [[HELLO_ADDR:x[0-9]+]], hello
// CHECK-NEXT: add [[HELLO_ADDR]], [[HELLO_ADDR]], :lo12:hello
// CHECK-NEXT: cmp x{{[0-9]+}}, [[HELLO_ADDR]]
// CHECK-NEXT: b.ne
// CHECK: bl    hello
// CHECK: End of Function "execute"
// CHECK: Binary Function "executeTailCall" after indirect-call-promotion
// CHECK: adrp [[EXIT_ADDR:x[0-9]+]], exit_success
// CHECK-NEXT: add [[EXIT_ADDR]], [[EXIT_ADDR]], :lo12:exit_success
// CHECK-NEXT: cmp x{{[0-9]+}}, [[EXIT_ADDR]]
// CHECK-NEXT: b.ne
// CHECK: b    exit_success
// CHECK: End of Function "executeTailCall"

typedef int (*IntCallback)(int);
typedef void (*VoidCallback)(void);

__attribute__((noinline)) void hello(void) {}

__attribute__((noinline)) void execute(VoidCallback Callback) { Callback(); }

// FDATA: 1 execute 14 1 hello 0 0 1

__attribute__((noinline)) int exit_success(int X) { return X; }

IntCallback TailCallback = exit_success;

__attribute__((noinline)) int executeTailCall(int X) {
  __attribute__((musttail)) return TailCallback(X);
}

// FDATA: 1 executeTailCall 18 1 exit_success 0 0 1

int main(void) {
  execute(hello);
  return executeTailCall(0);
}
