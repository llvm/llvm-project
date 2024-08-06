// Check various clang-linker-wrapper pass options after -offload-opt.

// REQUIRES: llvm-plugins, llvm-examples
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// https://github.com/llvm/llvm-project/issues/100212
// XFAIL: *

// Setup.
// RUN: mkdir -p %t
// RUN: %clang -cc1 -emit-llvm-bc -o %t/host-x86_64-unknown-linux-gnu.bc \
// RUN:     -disable-O0-optnone -triple=x86_64-unknown-linux-gnu %s
// RUN: %clang -cc1 -emit-llvm-bc -o %t/openmp-amdgcn-amd-amdhsa.bc \
// RUN:     -disable-O0-optnone -triple=amdgcn-amd-amdhsa %s
// RUN: opt %t/openmp-amdgcn-amd-amdhsa.bc -o %t/openmp-amdgcn-amd-amdhsa.bc \
// RUN:     -passes=forceattrs -force-remove-attribute=f:noinline
// RUN: clang-offload-packager -o %t/openmp-x86_64-unknown-linux-gnu.out \
// RUN:     --image=file=%t/openmp-amdgcn-amd-amdhsa.bc,arch=gfx90a,triple=amdgcn-amd-amdhsa
// RUN: %clang -cc1 -S -o %t/host-x86_64-unknown-linux-gnu.s \
// RUN:     -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:     -fembed-offload-object=%t/openmp-x86_64-unknown-linux-gnu.out \
// RUN:     %t/host-x86_64-unknown-linux-gnu.bc
// RUN: %clang -cc1as -o %t/host-x86_64-unknown-linux-gnu.o \
// RUN:     -triple x86_64-unknown-linux-gnu -filetype obj -target-cpu x86-64 \
// RUN:     %t/host-x86_64-unknown-linux-gnu.s

// Check plugin, -passes, and no remarks.
// RUN: clang-linker-wrapper -o a.out --embed-bitcode \
// RUN:     --linker-path=/usr/bin/true %t/host-x86_64-unknown-linux-gnu.o \
// RUN:     %offload-opt-loadbye --offload-opt=-wave-goodbye \
// RUN:     --offload-opt=-passes="function(goodbye),module(inline)" 2>&1 | \
// RUN:   FileCheck -match-full-lines -check-prefixes=OUT %s

// Check plugin, -p, and remarks.
// RUN: clang-linker-wrapper -o a.out --embed-bitcode \
// RUN:     --linker-path=/usr/bin/true %t/host-x86_64-unknown-linux-gnu.o \
// RUN:     %offload-opt-loadbye --offload-opt=-wave-goodbye \
// RUN:     --offload-opt=-p="function(goodbye),module(inline)" \
// RUN:     --offload-opt=-pass-remarks=inline \
// RUN:     --offload-opt=-pass-remarks-output=%t/remarks.yml \
// RUN:     --offload-opt=-pass-remarks-filter=inline \
// RUN:     --offload-opt=-pass-remarks-format=yaml 2>&1 | \
// RUN:   FileCheck -match-full-lines -check-prefixes=OUT,REM %s
// RUN: FileCheck -input-file=%t/remarks.yml -match-full-lines \
// RUN:     -check-prefixes=YML %s

// Check handling of bad plugin.
// RUN: not clang-linker-wrapper \
// RUN:     --offload-opt=-load-pass-plugin=%t/nonexistent.so 2>&1 | \
// RUN:   FileCheck -match-full-lines -check-prefixes=BAD-PLUGIN %s

//  OUT-NOT: {{.}}
//      OUT: Bye: f
// OUT-NEXT: Bye: test
// REM-NEXT: remark: {{.*}} 'f' inlined into 'test' {{.*}}
//  OUT-NOT: {{.}}

//  YML-NOT: {{.}}
//      YML: --- !Passed
// YML-NEXT: Pass: inline
// YML-NEXT: Name: Inlined
// YML-NEXT: Function: test
// YML-NEXT: Args:
//      YML:  - Callee: f
//      YML:  - Caller: test
//      YML: ...
//  YML-NOT: {{.}}

// BAD-PLUGIN-NOT: {{.}}
//     BAD-PLUGIN: {{.*}}Could not load library {{.*}}nonexistent.so{{.*}}
// BAD-PLUGIN-NOT: {{.}}

void f() {}
void test() { f(); }
