// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

#if defined(X)
extern int y;
int foo() { return y; }

int x = 0;
#elif defined(Y)
int y = 42;
#elif defined(Z)
int z = 42;
#elif defined(W)
int w = 42;
#elif defined(U)
extern int x;
extern int __attribute__((weak)) w;

int bar() {
  return x + w;
}
#else
extern int y;
extern int x;
int baz() { return y + x; }
#endif

// Create various inputs to test basic linking and LTO capabilities. Creating a
// CUDA binary requires access to the `ptxas` executable, so we just use x64.
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -o %t.o
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -DX -o %t-x.o
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -DY -o %t-y.o
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -DZ -o %t-z.o
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -DW -o %t-w.o
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -DU -o %t-u.o
// RUN: llvm-ar rcs %t-x.a %t-x.o
// RUN: llvm-ar rcs %t-y.a %t-y.o
// RUN: llvm-ar rcs %t-z.a %t-z.o
// RUN: llvm-ar rcs %t-w.a %t-w.o
// RUN: llvm-ar rcs %t-u.a %t-u.o

//
// Check that we forward any unrecognized argument to 'nvlink'.
//
// RUN: clang-nvlink-wrapper --dry-run -arch sm_52 %t-u.o -foo -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARGS
// ARGS: nvlink{{.*}} -arch sm_52 -foo -o a.out [[INPUT:.+]].cubin

//
// Check the symbol resolution for static archives. We expect to only link
// `libx.a` and `liby.a` because extern weak symbols do not extract and `libz.a`
// is not used at all.
//
// RUN: clang-nvlink-wrapper --dry-run %t-x.a %t-u.a %t-y.a %t-z.a %t-w.a %t.o \
// RUN:   -arch sm_52 -o a.out 2>&1 | FileCheck %s --check-prefix=LINK
// LINK: nvlink{{.*}} -arch sm_52 -o a.out [[INPUT:.+]].cubin {{.*}}-x-{{.*}}.cubin{{.*}}-y-{{.*}}.cubin

//
// Same as above but we use '--undefined' to forcibly extract 'libz.a'
//
// RUN: clang-nvlink-wrapper --dry-run %t-x.a %t-u.a %t-y.a %t-z.a %t-w.a %t.o \
// RUN:   -u z -arch sm_52 -o a.out 2>&1 | FileCheck %s --check-prefix=LINK
// UNDEFINED: nvlink{{.*}} -arch sm_52 -o a.out [[INPUT:.+]].cubin {{.*}}-x-{{.*}}.cubin{{.*}}-y-{{.*}}.cubin{{.*}}-z-{{.*}}.cubin

//
// Check that the LTO interface works and properly preserves symbols used in a
// regular object file.
//
// RUN: clang-nvlink-wrapper --dry-run %t.o %t-u.o %t-y.a \
// RUN:   -arch sm_52 -o a.out 2>&1 | FileCheck %s --check-prefix=LTO
// LTO: ptxas{{.*}} -m64 -c [[PTX:.+]].s -O3 -arch sm_52 -o [[CUBIN:.+]].cubin
// LTO: nvlink{{.*}} -arch sm_52 -o a.out [[CUBIN]].cubin {{.*}}-u-{{.*}}.cubin {{.*}}-y-{{.*}}.cubin

//
// Check that we don't forward some arguments.
//
// RUN: clang-nvlink-wrapper --dry-run %t.o %t-u.o %t-y.a \
// RUN:   -arch sm_52 --cuda-path/opt/cuda -o a.out 2>&1 | FileCheck %s --check-prefix=PATH
// PATH-NOT: --cuda-path=/opt/cuda

//
// Check that passes can be specified and debugged.
//
// RUN: clang-nvlink-wrapper --dry-run %t.o %t-u.o %t-y.a \
// RUN:   --lto-debug-pass-manager --lto-newpm-passes=forceattrs \
// RUN:   -arch sm_52 -o a.out 2>&1 | FileCheck %s --check-prefix=PASSES
// PASSES: Running pass: ForceFunctionAttrsPass

//
// Check that '-plugin` is ingored like in `ld.lld`
//
// RUN: clang-nvlink-wrapper --dry-run %t.o -plugin foo.so -arch sm_52 -o a.out \
// RUN:   2>&1 | FileCheck %s --check-prefix=PLUGIN
// PLUGIN-NOT: -plugin
