// REQUIRES: x86-registered-target

//
// Check help message.
//
// RUN: clang-offload-wrapper --help | FileCheck %s --check-prefix CHECK-HELP
// CHECK-HELP: {{.*}}OVERVIEW: A tool to create a wrapper bitcode for offload target binaries. Takes offload
// CHECK-HELP: {{.*}}target binaries as input and produces bitcode file containing target binaries packaged
// CHECK-HELP: {{.*}}as data.
// CHECK-HELP: {{.*}}USAGE: clang-offload-wrapper [options] <input files>
// CHECK-HELP: {{.*}}  -o=<filename>               - Output filename
// CHECK-HELP: {{.*}}  --offload-targets=<triples> - Comma-separated list of device target triples
// CHECK-HELP: {{.*}}  --target=<triple>           - Target triple for the output module

//
// Generate a file to wrap.
//
// RUN: echo 'Content of device file' > %t.tgt

//
// Check bitcode produced by the wrapper tool.
//
// RUN: clang-offload-wrapper -target=x86_64-pc-linux-gnu -offload-targets=powerpc64le-ibm-linux-gnu -o %t.wrapper.bc %t.tgt
// RUN: llvm-dis %t.wrapper.bc -o - | FileCheck %s --check-prefix CHECK-IR

// CHECK-IR: target triple = "x86_64-pc-linux-gnu"

// CHECK-IR: @.omp_offloading.img_start.powerpc64le-ibm-linux-gnu = hidden unnamed_addr constant [{{[0-9]+}} x i8] c"Content of device file{{.+}}", section ".omp_offloading.powerpc64le-ibm-linux-gnu"
// CHECK-IR: @.omp_offloading.img_end.powerpc64le-ibm-linux-gnu = hidden unnamed_addr constant [0 x i8] zeroinitializer, section ".omp_offloading.powerpc64le-ibm-linux-gnu"
