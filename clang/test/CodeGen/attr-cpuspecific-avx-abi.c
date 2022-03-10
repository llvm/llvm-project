// RUN: %clang_cc1 -triple x86_64-linux-gnu -target-feature +avx -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK

// Make sure the features from the command line are honored regardless of what
// CPU is specified in the cpu_specific attribute.
// In this test, if the 'avx' feature isn't honored, we'll generate an error for
// the return type having a different ABI without 'avx' being enabled.

typedef double __m256d __attribute__((vector_size(32)));

extern __m256d bar_avx1(void);
extern __m256d bar_avx2(void);

// AVX1/AVX2 dispatcher
__attribute__((cpu_dispatch(generic, core_4th_gen_avx)))
__m256d foo_pd64x4(void);

__attribute__((cpu_specific(generic)))
__m256d foo(void) { return bar_avx1(); }
// CHECK: define{{.*}} @foo.A() #[[A:[0-9]+]]

__attribute__((cpu_specific(core_4th_gen_avx)))
__m256d foo(void) { return bar_avx2(); }
// CHECK: define{{.*}} @foo.V() #[[V:[0-9]+]]

// CHECK: attributes #[[A]] = {{.*}}"target-features"="+avx,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
// CHECK-SAME: "tune-cpu"="generic"
// CHECK: attributes #[[V]] = {{.*}}"target-features"="+avx,+avx2,+bmi,+cmov,+crc32,+cx8,+f16c,+fma,+lzcnt,+mmx,+movbe,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
// CHECK-SAME: "tune-cpu"="haswell"
