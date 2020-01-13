// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -o %t.o -flto=thin -fexperimental-new-pass-manager -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s
// RUN: llvm-lto -thinlto -o %t %t.o

// Test to ensure the slp vectorize codegen option is passed down to the
// ThinLTO backend. -vectorize-slp is a cc1 option and will be added
// automatically when O2/O3/Os/Oz is available for clang. Once -vectorize-slp
// is enabled, "-mllvm -vectorize-slp=false" won't disable slp vectorization
// currently. "-mllvm -vectorize-slp=false" is added here in the test to
// ensure the slp vectorization is executed because the -vectorize-slp cc1
// flag is passed down, not because "-mllvm -vectorize-slp" is enabled
// by default.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -vectorize-slp -mllvm -vectorize-slp=false -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=O2-SLP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O0 -vectorize-slp -mllvm -vectorize-slp=false -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=O0-SLP
// O2-SLP: Running pass: SLPVectorizerPass
// O0-SLP-NOT: Running pass: SLPVectorizerPass

// Test to ensure the loop vectorize codegen option is passed down to the
// ThinLTO backend. -vectorize-loops is a cc1 option and will be added
// automatically when O2/O3/Os is available for clang. Once -vectorize-loops is
// enabled, "-mllvm -vectorize-loops=false" won't disable loop vectorization
// currently. "-mllvm -vectorize-loops=false" is added here in the test to
// ensure the loop vectorization is executed because the -vectorize-loops cc1
// flag is passed down, not because "-mllvm -vectorize-loops" is enabled
// by default.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -vectorize-loops -mllvm -vectorize-loops=false -mllvm -force-vector-width=2 -mllvm -force-vector-interleave=1 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=O2-LPV
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O0 -vectorize-loops -mllvm -vectorize-loops=false -mllvm -force-vector-width=2 -mllvm -force-vector-interleave=1 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=O0-LPV
// O2-LPV: = !{!"llvm.loop.isvectorized", i32 1}
// O0-LPV-NOT: = !{!"llvm.loop.isvectorized", i32 1}

// Test to ensure the loop interleave codegen option is passed down to the
// ThinLTO backend. The internal loop interleave codegen option will be
// enabled automatically when O2/O3 is available for clang. Once the loop
// interleave option is enabled, "-mllvm -interleave-loops=false" won't disable
// the interleave. currently. "-mllvm -interleave-loops=false" is added here
// in the test to ensure the loop interleave is executed because the interleave
// codegen flag is passed down, not because "-mllvm -interleave-loops" is
// enabled by default.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -vectorize-loops -mllvm -interleave-loops=false -mllvm -force-vector-width=1 -mllvm -force-vector-interleave=2 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=O2-InterLeave
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O0 -vectorize-loops -mllvm -interleave-loops=false -mllvm -force-vector-width=1 -mllvm -force-vector-interleave=2 -emit-llvm -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=O0-InterLeave
// O2-InterLeave: = !{!"llvm.loop.isvectorized", i32 1}
// O0-InterLeave-NOT: = !{!"llvm.loop.isvectorized", i32 1}

void foo(double *a) {
  for (int i = 0; i < 1000; i++)
    a[i] = 10;
}
