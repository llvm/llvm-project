// REQUIRES: zstd
// REQUIRES: x86-registered-target
// UNSUPPORTED: target={{.*}}-darwin{{.*}}, target={{.*}}-aix{{.*}}

//
// Generate the host binary to be bundled.
//
// RUN: %clang -O0 -target %itanium_abi_triple %s -c -emit-llvm -o %t.bc

//
// Generate an empty file to help with the checks of empty files.
//
// RUN: touch %t.empty

//
// Generate device binaries to be bundled.
//
// RUN: echo 'Content of device file 1' > %t.tgt1
// RUN: echo 'Content of device file 2' > %t.tgt2

//
// Check compression/decompression of offload bundle.
//
// RUN: clang-offload-bundler -type=bc -targets=hip-amdgcn-amd-amdhsa--gfx900,hip-amdgcn-amd-amdhsa--gfx906 \
// RUN:   -input=%t.tgt1 -input=%t.tgt2 -output=%t.hip.bundle.bc -compress -verbose 2>&1 | \
// RUN:   FileCheck -check-prefix=COMPRESS %s
// RUN: clang-offload-bundler -type=bc -list -input=%t.hip.bundle.bc | FileCheck -check-prefix=NOHOST %s
// RUN: clang-offload-bundler -type=bc -targets=hip-amdgcn-amd-amdhsa--gfx900,hip-amdgcn-amd-amdhsa--gfx906 \
// RUN:   -output=%t.res.tgt1 -output=%t.res.tgt2 -input=%t.hip.bundle.bc -unbundle -verbose 2>&1 | \
// RUN:   FileCheck -check-prefix=DECOMPRESS %s
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
//
// COMPRESS: Compression method used
// DECOMPRESS: Decompression method
// NOHOST-NOT: host-
// NOHOST-DAG: hip-amdgcn-amd-amdhsa--gfx900
// NOHOST-DAG: hip-amdgcn-amd-amdhsa--gfx906
//

//
// Check -bundle-align option.
//

// RUN: clang-offload-bundler -bundle-align=4096 -type=bc -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -input=%t.bc -input=%t.tgt1 -input=%t.tgt2 -output=%t.bundle3.bc -compress
// RUN: clang-offload-bundler -type=bc -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -output=%t.res.bc -output=%t.res.tgt1 -output=%t.res.tgt2 -input=%t.bundle3.bc -unbundle
// RUN: diff %t.bc %t.res.bc
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2

//
// Check unbundling archive.
//
// RUN: clang-offload-bundler -type=bc -targets=hip-amdgcn-amd-amdhsa--gfx900,hip-amdgcn-amd-amdhsa--gfx906 \
// RUN:   -input=%t.tgt1 -input=%t.tgt2 -output=%T/hip_bundle1.bc -compress
// RUN: clang-offload-bundler -type=bc -targets=hip-amdgcn-amd-amdhsa--gfx900,hip-amdgcn-amd-amdhsa--gfx906 \
// RUN:   -input=%t.tgt1 -input=%t.tgt2 -output=%T/hip_bundle2.bc -compress
// RUN: llvm-ar cr %T/hip_archive.a %T/hip_bundle1.bc %T/hip_bundle2.bc
// RUN: clang-offload-bundler -unbundle -type=a -targets=hip-amdgcn-amd-amdhsa--gfx900,hip-amdgcn-amd-amdhsa--gfx906 \
// RUN:   -output=%T/hip_900.a -output=%T/hip_906.a -input=%T/hip_archive.a
// RUN: llvm-ar t %T/hip_900.a | FileCheck -check-prefix=HIP-AR-900 %s
// RUN: llvm-ar t %T/hip_906.a | FileCheck -check-prefix=HIP-AR-906 %s
// HIP-AR-900-DAG: hip_bundle1-hip-amdgcn-amd-amdhsa--gfx900
// HIP-AR-900-DAG: hip_bundle2-hip-amdgcn-amd-amdhsa--gfx900
// HIP-AR-906-DAG: hip_bundle1-hip-amdgcn-amd-amdhsa--gfx906
// HIP-AR-906-DAG: hip_bundle2-hip-amdgcn-amd-amdhsa--gfx906

// Some code so that we can create a binary out of this file.
int A = 0;
void test_func(void) {
  ++A;
}
