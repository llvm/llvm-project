// This test verifies inlining after indirect call promotion on AArch64.

// RUN: %clang %cflags -O0 %s -o %t.exe -Wl,-q
// RUN: link_fdata %s %t.exe %t.fdata
// RUN: llvm-bolt %t.exe --icp=calls --icp-calls-topn=1 \
// RUN:   --inline-small-functions -o %t.null --lite=0 --assume-abi \
// RUN:   --inline-small-functions-bytes=32 --print-inline --data %t.fdata \
// RUN:   | FileCheck %s

// CHECK:     Binary Function "indirectTailCall" after inlining
// CHECK:     adrp [[BAR_ADDR:x[0-9]+]], bar
// CHECK-NEXT: add [[BAR_ADDR]], [[BAR_ADDR]], :lo12:bar
// CHECK-NEXT: cmp x{{[0-9]+}}, [[BAR_ADDR]]
// CHECK-NEXT: b.ne
// The multiply comes from bar's body, confirming it was inlined after ICP.
// CHECK:     mul     w8, w8, w9
// CHECK-NOT: b    bar
// CHECK:     br    x{{[0-9]+}}
// CHECK:     End of Function "indirectTailCall"

__attribute__((noinline)) int foo(int X) { return X + 1; }
__attribute__((noinline)) int bar(int X) { return X * 100 + 42; }

typedef int (*Fn)(int);

Fn funcs[] = {foo, bar};

__attribute__((noinline)) int indirectTailCall(int Argc) {
  Fn Func = funcs[Argc];
  __attribute__((musttail)) return Func(0);
}

// FDATA: 1 indirectTailCall 28 1 foo 0 0 1
// FDATA: 1 indirectTailCall 28 1 bar 0 0 2

int main(int Argc, char **Argv) {
  (void)Argv;
  return indirectTailCall(Argc & 1);
}
