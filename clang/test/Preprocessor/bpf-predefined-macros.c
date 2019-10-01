// RUN: %clang -E -target bpfel -x c -o - %s | FileCheck %s
// RUN: %clang -E -target bpfeb -x c -o - %s | FileCheck %s

#ifdef __bpf__
int b;
#endif
#ifdef __BPF__
int c;
#endif
#ifdef bpf
int d;
#endif

// CHECK: int b;
// CHECK: int c;
// CHECK-NOT: int d;
