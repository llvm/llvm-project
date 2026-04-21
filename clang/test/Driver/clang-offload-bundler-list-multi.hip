// clang-offload-bundler-list-multi.c
  // REQUIRES: x86-registered-target
  // UNSUPPORTED: target={{.*}}-macosx{{.*}}, target={{.*}}-darwin{{.*}}, target={{.*}}-aix{{.*}}, target={{.*}}-zos{{.*}}

  // Test that --list works on shared objects and archives that contain multiple
  // offload fat binary bundles (i.e., multiple bundled object members).

  //
  // Setup: build two simple source files and produce bundled objects for them.
  //
  // RUN: echo 'void foo() {}' > %t.foo.c
  // RUN: echo 'void bar() {}' > %t.bar.c
  // RUN: %clang -target %itanium_abi_triple %t.foo.c -c -o %t.foo.o
  // RUN: %clang -target %itanium_abi_triple %t.bar.c -c -o %t.bar.o
  // RUN: echo 'device content 1' > %t.dev1
  // RUN: echo 'device content 2' > %t.dev2

  //
  // Bundle each host object together with two HIP device targets.
  //
  // RUN: clang-offload-bundler -type=o \
  // RUN:   -targets=host-%itanium_abi_triple,hip-amdgcn-amd-amdhsa--gfx906,hip-amdgcn-amd-amdhsa--gfx908 \
  // RUN:   -input=%t.foo.o -input=%t.dev1 -input=%t.dev2 \
  // RUN:   -output=%t.foo.bundled.o
  // RUN: clang-offload-bundler -type=o \
  // RUN:   -targets=host-%itanium_abi_triple,hip-amdgcn-amd-amdhsa--gfx906,hip-amdgcn-amd-amdhsa--gfx908 \
  // RUN:   -input=%t.bar.o -input=%t.dev1 -input=%t.dev2 \
  // RUN:   -output=%t.bar.bundled.o

  //
  // Check --list on a single bundled object (baseline).
  //
  // RUN: clang-offload-bundler -type=o -list -input=%t.foo.bundled.o \
  // RUN:   | FileCheck %s --check-prefix=LIST-OBJ
  // LIST-OBJ-DAG: host-
  // LIST-OBJ-DAG: hip-amdgcn-amd-amdhsa--gfx906
  // LIST-OBJ-DAG: hip-amdgcn-amd-amdhsa--gfx908

  //
  // Create a shared object from the two bundled objects, then check --list on it.
  // The shared object retains the offload bundle ELF sections, so --list should
  // report the union of all bundle IDs present.
  //
  // RUN: %clang -target %itanium_abi_triple -shared \
  // RUN:   %t.foo.bundled.o %t.bar.bundled.o -o %t.multi.so
  // RUN: clang-offload-bundler -type=o -list -input=%t.multi.so \
  // RUN:   | FileCheck %s --check-prefix=LIST-SO
  // LIST-SO-DAG: host-
  // LIST-SO-DAG: hip-amdgcn-amd-amdhsa--gfx906
  // LIST-SO-DAG: hip-amdgcn-amd-amdhsa--gfx908

  //
  // Create an archive containing multiple bundled objects and check --list.
  // (Archives use -type=a; --list is supported for archives as for objects.)
  //
  // RUN: rm -f %t.multi.a
  // RUN: llvm-ar cr %t.multi.a %t.foo.bundled.o %t.bar.bundled.o
  // RUN: clang-offload-bundler -type=a -list -input=%t.multi.a \
  // RUN:   | FileCheck %s --check-prefix=LIST-AR
  // LIST-AR-DAG: host-
  // LIST-AR-DAG: hip-amdgcn-amd-amdhsa--gfx906
  // LIST-AR-DAG: hip-amdgcn-amd-amdhsa--gfx908

  // Some code so we can compile this file.
__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
