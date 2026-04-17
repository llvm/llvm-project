// REQUIRES: x86-registered-target
// REQUIRES: zlib || zstd
// UNSUPPORTED: target={{.*}}-darwin{{.*}}, target={{.*}}-aix{{.*}}, target={{.*}}-zos{{.*}}

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
// RUN: %clang -O0 -target %itanium_abi_triple %s -c -o %t.host.o
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

// Some code so that we can compile this file as a host object.
int A = 0;
void test_func(void) { ++A; }

