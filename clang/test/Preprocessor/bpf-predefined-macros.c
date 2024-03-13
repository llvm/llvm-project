// RUN: %clang -E -target bpfel -x c -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CPU_NO %s
// RUN: %clang -E -target bpfeb -x c -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CPU_NO %s
// RUN: %clang -E -target bpfel -mcpu=v1 -x c -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CPU_V1 %s
// RUN: %clang -E -target bpfel -mcpu=v2 -x c -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CPU_V2 %s
// RUN: %clang -E -target bpfel -mcpu=v3 -x c -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CPU_V3 %s
// RUN: %clang -E -target bpfel -mcpu=v4 -x c -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CPU_V4 %s
// RUN: %clang -E -target bpfel -mcpu=generic -x c -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CPU_GENERIC %s
// RUN: %clang -E -target bpfel -mcpu=probe -x c -o - %s | FileCheck -check-prefix=CHECK -check-prefix=CPU_PROBE %s

#ifdef __bpf__
int b;
#endif
#ifdef __BPF__
int c;
#endif
#ifdef bpf
int d;
#endif
#ifdef __BPF_CPU_VERSION__
int e;
#endif
#if __BPF_CPU_VERSION__ == 0
int f;
#endif
#if __BPF_CPU_VERSION__ == 1
int g;
#endif
#if __BPF_CPU_VERSION__ == 2
int h;
#endif
#if __BPF_CPU_VERSION__ == 3
int i;
#endif
#if __BPF_CPU_VERSION__ == 4
int j;
#endif
#ifdef __BPF_FEATURE_JMP_EXT
int k;
#endif
#ifdef __BPF_FEATURE_JMP32
int l;
#endif
#ifdef __BPF_FEATURE_ALU32
int m;
#endif
#ifdef __BPF_FEATURE_LDSX
int n;
#endif
#ifdef __BPF_FEATURE_MOVSX
int o;
#endif
#ifdef __BPF_FEATURE_BSWAP
int p;
#endif
#ifdef __BPF_FEATURE_SDIV_SMOD
int q;
#endif
#ifdef __BPF_FEATURE_GOTOL
int r;
#endif
#ifdef __BPF_FEATURE_ST
int s;
#endif
#ifdef __BPF_FEATURE_ADDRESS_SPACE_CAST
int t;
#endif

// CHECK: int b;
// CHECK: int c;
// CHECK-NOT: int d;
// CHECK: int e;

// CPU_NO: int g;

// CPU_V1: int g;

// CPU_V2: int h;
// CPU_V2: int k;

// CPU_V3: int i;
// CPU_V3: int k;
// CPU_V3: int l;
// CPU_V3: int m;

// CPU_V4: int j;
// CPU_V4: int k;
// CPU_V4: int l;
// CPU_V4: int m;
// CPU_V4: int n;
// CPU_V4: int o;
// CPU_V4: int p;
// CPU_V4: int q;
// CPU_V4: int r;
// CPU_V4: int s;

// CPU_V1: int t;
// CPU_V2: int t;
// CPU_V3: int t;
// CPU_V4: int t;

// CPU_GENERIC: int g;

// CPU_PROBE: int f;
