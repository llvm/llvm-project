// REQUIRES: x86-registered-target
// REQUIRES: zlib || zstd

// Tests that clang-offload-bundler --list correctly enumerates all bundle IDs
// from multiple concatenated compressed (CCOB) fat binary blobs stored in the
// .hip_fatbin section of an ELF object. This models the layout produced when
// a shared library or relocatable object is linked from multiple HIP
// translation units, each of which contributes its own CCOB blob to the
// .hip_fatbin section.

//
// Create device content files for two simulated translation units.
//
// RUN: echo 'Content of device file 1' > %t.dev1
// RUN: echo 'Content of device file 2' > %t.dev2
// RUN: echo 'Content of device file 3' > %t.dev3

//
// Produce two compressed fat binary blobs with distinct GPU targets so that
// the FileCheck assertions below are unambiguous.
//
// Bundle 1: gfx906 + gfx908
// RUN: clang-offload-bundler -compress -type=bc \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx906,hip-amdgcn-amd-amdhsa--gfx908 \
// RUN:   -input=%t.dev1 -input=%t.dev2 \
// RUN:   -output=%t.bundle1.ccob

// Bundle 2: gfx1030 + gfx1100
// RUN: clang-offload-bundler -compress -type=bc \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx1030,hip-amdgcn-amd-amdhsa--gfx1100 \
// RUN:   -input=%t.dev1 -input=%t.dev2 \
// RUN:   -output=%t.bundle2.ccob

// Bundle 3: gfx942 + gfx1201
// RUN: clang-offload-bundler -compress -type=bc \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx942,hip-amdgcn-amd-amdhsa--gfx1201 \
// RUN:   -input=%t.dev1 -input=%t.dev3 \
// RUN:   -output=%t.bundle3.ccob

//
// Baseline: --list on each individual compressed bundle must work.
//
// RUN: clang-offload-bundler -type=bc -list -input=%t.bundle1.ccob \
// RUN:   | FileCheck %s --check-prefix=SINGLE1
// SINGLE1-DAG: hip-amdgcn-amd-amdhsa--gfx906
// SINGLE1-DAG: hip-amdgcn-amd-amdhsa--gfx908

// RUN: clang-offload-bundler -type=bc -list -input=%t.bundle2.ccob \
// RUN:   | FileCheck %s --check-prefix=SINGLE2
// SINGLE2-DAG: hip-amdgcn-amd-amdhsa--gfx1030
// SINGLE2-DAG: hip-amdgcn-amd-amdhsa--gfx1100

//
// Concatenate the two CCOB blobs. This mirrors what the linker does when it
// merges the .hip_fatbin input sections contributed by multiple TUs.
//
// RUN: cat %t.bundle1.ccob %t.bundle2.ccob > %t.multi.fatbin

//
// Build a host object and inject the concatenated blob as its .hip_fatbin
// section, replicating the ELF layout of a fat shared library.
//
// RUN: %clang -O0 -target x86_64-unknown-linux-gnu %s -c -o %t.host.o
// RUN: llvm-objcopy \
// RUN:   --add-section=.hip_fatbin=%t.multi.fatbin \
// RUN:   --set-section-flags=.hip_fatbin=alloc \
// RUN:   %t.host.o %t.multi.o

//
// --list on an object whose .hip_fatbin section contains two concatenated
// CCOB blobs must enumerate all bundle IDs from both fat binaries.
//
// RUN: clang-offload-bundler -type=o -list -input=%t.multi.fatbin \
// RUN:   | FileCheck %s --check-prefix=MULTI
// MULTI-DAG: hip-amdgcn-amd-amdhsa--gfx906
// MULTI-DAG: hip-amdgcn-amd-amdhsa--gfx908
// MULTI-DAG: hip-amdgcn-amd-amdhsa--gfx1030
// MULTI-DAG: hip-amdgcn-amd-amdhsa--gfx1100

//
// Concatenate three CCOB blobs to verify that the loop correctly processes
// 3+ blobs without premature termination.
//
// RUN: cat %t.bundle1.ccob %t.bundle2.ccob %t.bundle3.ccob > %t.triple.fatbin

// --list on three concatenated CCOB blobs must enumerate all bundle IDs.
// RUN: clang-offload-bundler -type=o -list -input=%t.triple.fatbin \
// RUN:   | FileCheck %s --check-prefix=TRIPLE
// TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx906
// TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx908
// TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx1030
// TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx1100
// TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx942
// TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx1201

// --unbundle must extract targets spanning all three CCOB blobs.
// RUN: clang-offload-bundler -type=o -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx906,hip-amdgcn-amd-amdhsa--gfx1030,hip-amdgcn-amd-amdhsa--gfx942 \
// RUN:   -output=%t.tri.res.gfx906 -output=%t.tri.res.gfx1030 -output=%t.tri.res.gfx942 \
// RUN:   -input=%t.triple.fatbin
// RUN: diff %t.dev1 %t.tri.res.gfx906
// RUN: diff %t.dev1 %t.tri.res.gfx1030
// RUN: diff %t.dev1 %t.tri.res.gfx942

//
// ===--- Uncompressed multi-bundle tests ---===
//
// Repeat the same --list and --unbundle tests using uncompressed fat binary
// blobs (__CLANG_OFFLOAD_BUNDLE__ binary format without CCOB).
//

// Bundle 1 (uncompressed): gfx906 + gfx908
// RUN: clang-offload-bundler -type=bc \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx906,hip-amdgcn-amd-amdhsa--gfx908 \
// RUN:   -input=%t.dev1 -input=%t.dev2 \
// RUN:   -output=%t.unc.bundle1.bc

// Bundle 2 (uncompressed): gfx1030 + gfx1100
// RUN: clang-offload-bundler -type=bc \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx1030,hip-amdgcn-amd-amdhsa--gfx1100 \
// RUN:   -input=%t.dev1 -input=%t.dev2 \
// RUN:   -output=%t.unc.bundle2.bc

// Bundle 3 (uncompressed): gfx942 + gfx1201
// RUN: clang-offload-bundler -type=bc \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx942,hip-amdgcn-amd-amdhsa--gfx1201 \
// RUN:   -input=%t.dev1 -input=%t.dev3 \
// RUN:   -output=%t.unc.bundle3.bc

// Concatenate the two uncompressed blobs.
// RUN: cat %t.unc.bundle1.bc %t.unc.bundle2.bc > %t.unc.multi.fatbin

// --list must enumerate all bundle IDs from both uncompressed blobs.
// RUN: clang-offload-bundler -type=o -list -input=%t.unc.multi.fatbin \
// RUN:   | FileCheck %s --check-prefix=UNC-MULTI
// UNC-MULTI-DAG: hip-amdgcn-amd-amdhsa--gfx906
// UNC-MULTI-DAG: hip-amdgcn-amd-amdhsa--gfx908
// UNC-MULTI-DAG: hip-amdgcn-amd-amdhsa--gfx1030
// UNC-MULTI-DAG: hip-amdgcn-amd-amdhsa--gfx1100

// --unbundle must extract targets spanning both uncompressed blobs.
// RUN: clang-offload-bundler -type=o -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx906,hip-amdgcn-amd-amdhsa--gfx1100 \
// RUN:   -output=%t.unc.res.gfx906 -output=%t.unc.res.gfx1100 \
// RUN:   -input=%t.unc.multi.fatbin
// RUN: diff %t.dev1 %t.unc.res.gfx906
// RUN: diff %t.dev2 %t.unc.res.gfx1100

// Concatenate three uncompressed blobs.
// RUN: cat %t.unc.bundle1.bc %t.unc.bundle2.bc %t.unc.bundle3.bc > %t.unc.triple.fatbin

// --list on three concatenated uncompressed blobs must enumerate all bundle IDs.
// RUN: clang-offload-bundler -type=o -list -input=%t.unc.triple.fatbin \
// RUN:   | FileCheck %s --check-prefix=UNC-TRIPLE
// UNC-TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx906
// UNC-TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx908
// UNC-TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx1030
// UNC-TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx1100
// UNC-TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx942
// UNC-TRIPLE-DAG: hip-amdgcn-amd-amdhsa--gfx1201

// --unbundle must extract targets spanning all three uncompressed blobs.
// RUN: clang-offload-bundler -type=o -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx906,hip-amdgcn-amd-amdhsa--gfx1030,hip-amdgcn-amd-amdhsa--gfx942 \
// RUN:   -output=%t.unc.tri.res.gfx906 -output=%t.unc.tri.res.gfx1030 -output=%t.unc.tri.res.gfx942 \
// RUN:   -input=%t.unc.triple.fatbin
// RUN: diff %t.dev1 %t.unc.tri.res.gfx906
// RUN: diff %t.dev1 %t.unc.tri.res.gfx1030
// RUN: diff %t.dev1 %t.unc.tri.res.gfx942

//
// --unbundle on the same concatenated CCOB file must correctly extract targets
// that span both blobs in a single call. gfx906 comes from bundle1 and gfx1100
// comes from bundle2, so this exercises cross-blob extraction.
//
// RUN: clang-offload-bundler -type=o -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa--gfx906,hip-amdgcn-amd-amdhsa--gfx1100 \
// RUN:   -output=%t.res.gfx906 -output=%t.res.gfx1100 \
// RUN:   -input=%t.multi.fatbin
// RUN: diff %t.dev1 %t.res.gfx906
// RUN: diff %t.dev2 %t.res.gfx1100

// Some code so that we can compile this file as a host object.
int A = 0;
void test_func(void) { ++A; }

