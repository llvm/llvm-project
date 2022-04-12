// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -o %t.o -O2 -flto=thin -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s
// RUN: llvm-lto -thinlto -o %t %t.o

// Test to ensure the loop vectorize codegen option is passed down to the
// ThinLTO backend. -vectorize-loops is a cc1 option and will be added
// automatically when O2/O3/Os is available for clang. Also check that
// "-mllvm -vectorize-loops=false" will disable loop vectorization, overriding
// the cc1 option.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -vectorize-loops -mllvm -force-vector-width=2 -mllvm -force-vector-interleave=1 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s --check-prefix=O2-LPV
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -vectorize-loops -mllvm -vectorize-loops=false -mllvm -force-vector-width=2 -mllvm -force-vector-interleave=1 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s --check-prefix=O2-NOLPV
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O0 -vectorize-loops -mllvm -force-vector-width=2 -mllvm -force-vector-interleave=1 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s --check-prefix=O0-LPV
// O2-LPV: = !{!"llvm.loop.isvectorized", i32 1}
// O2-NOLPV-NOT: = !{!"llvm.loop.isvectorized", i32 1}
// O0-LPV-NOT: = !{!"llvm.loop.isvectorized", i32 1}

// Test to ensure the loop interleave codegen option is passed down to the
// ThinLTO backend. The internal loop interleave codegen option will be
// enabled automatically when O2/O3 is available for clang. Also check that
// "-mllvm -interleave-loops=false" will disable the interleaving, overriding
// the cc1 option.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -vectorize-loops -mllvm -force-vector-width=2 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s --check-prefix=O2-InterLeave
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -vectorize-loops -mllvm -interleave-loops=false -mllvm -force-vector-width=2 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s --check-prefix=O2-NoInterLeave
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O0 -vectorize-loops -mllvm -force-vector-width=2 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s --check-prefix=O0-InterLeave
// O2-InterLeave-COUNT-2: store <2 x double>
// O2-InterLeave: = !{!"llvm.loop.isvectorized", i32 1}
// O2-NoInterLeave-COUNT-1: store <2 x double>
// O2-NoInterLeave-NOT: store <2 x double>
// O2-NoInterLeave: = !{!"llvm.loop.isvectorized", i32 1}
// O0-InterLeave-NOT: = !{!"llvm.loop.isvectorized", i32 1}

void foo(double *a) {
  for (int i = 0; i < 1000; i++)
    a[i] = 10;
}
