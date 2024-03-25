// REQUIRES: x86-registered-target
// REQUIRES: asserts
// UNSUPPORTED: target={{.*}}-macosx{{.*}}, target={{.*}}-darwin{{.*}}, target={{.*}}-aix{{.*}}

// Generate the file we can bundle.
// RUN: %clang -O0 -target %itanium_abi_triple %s -c -o %t.o

//
// Generate a couple of files to bundle with.
//
// RUN: echo 'Content of device file 1' > %t.tgt1
// RUN: echo 'Content of device file 2' > %t.tgt2

//
// Check code object compatibility for archive unbundling
//
// Create few code object bundles and archive them to create an input archive
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-amdgcn-amd-amdhsa-gfx906,openmp-amdgcn-amd-amdhsa--gfx908 -input=%t.o -input=%t.tgt1 -input=%t.tgt2 -output=%t.simple.bundle
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-amdgcn-amd-amdhsa--gfx906:sramecc+:xnack+,openmp-amdgcn-amd-amdhsa--gfx908:sramecc+:xnack+ -inputs=%t.o,%t.tgt1,%t.tgt1 -outputs=%t.targetID1.bundle
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-amdgcn-amd-amdhsa--gfx906:sramecc+:xnack-,openmp-amdgcn-amd-amdhsa--gfx908:sramecc+:xnack- -inputs=%t.o,%t.tgt1,%t.tgt1 -outputs=%t.targetID2.bundle
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-amdgcn-amd-amdhsa--gfx906:xnack-,openmp-amdgcn-amd-amdhsa--gfx908:xnack- -inputs=%t.o,%t.tgt1,%t.tgt1 -outputs=%t.targetID3.bundle
// RUN: llvm-ar cr %t.input-archive.a %t.simple.bundle %t.targetID1.bundle %t.targetID2.bundle %t.targetID3.bundle

// Tests to check compatibility between Bundle Entry ID formats i.e. between presence/absence of extra hyphen in case of missing environment field
// RUN: clang-offload-bundler -unbundle -type=a -targets=openmp-amdgcn-amd-amdhsa--gfx906,openmp-amdgcn-amd-amdhsa-gfx908 -input=%t.input-archive.a -output=%t-archive-gfx906-simple.a -output=%t-archive-gfx908-simple.a -debug-only=CodeObjectCompatibility 2>&1 | FileCheck %s -check-prefix=BUNDLECOMPATIBILITY
// BUNDLECOMPATIBILITY: Compatible: Exact match:        [CodeObject: openmp-amdgcn-amd-amdhsa--gfx906]  :       [Target: openmp-amdgcn-amd-amdhsa--gfx906]
// BUNDLECOMPATIBILITY: Compatible: Exact match:        [CodeObject: openmp-amdgcn-amd-amdhsa--gfx908]  :       [Target: openmp-amdgcn-amd-amdhsa--gfx908]

// RUN: clang-offload-bundler -unbundle -type=a -targets=hip-amdgcn-amd-amdhsa--gfx906,hipv4-amdgcn-amd-amdhsa-gfx908 -input=%t.input-archive.a -output=%t-hip-archive-gfx906-simple.a -output=%t-hipv4-archive-gfx908-simple.a -hip-openmp-compatible -debug-only=CodeObjectCompatibility 2>&1 | FileCheck %s -check-prefix=HIPOpenMPCOMPATIBILITY
// HIPOpenMPCOMPATIBILITY: Compatible: Target IDs are compatible        [CodeObject: openmp-amdgcn-amd-amdhsa--gfx906]  :       [Target: hip-amdgcn-amd-amdhsa--gfx906]
// HIPOpenMPCOMPATIBILITY: Compatible: Target IDs are compatible        [CodeObject: openmp-amdgcn-amd-amdhsa--gfx908]  :       [Target: hipv4-amdgcn-amd-amdhsa--gfx908]

// Some code so that we can create a binary out of this file.
int A = 0;
void test_func(void) {
  ++A;
}
