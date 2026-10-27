// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -target-cpu pwr7 -emit-llvm -o - %s | FileCheck %s

// The 'target' attribute is not used for function multiversioning on AIX,
// which uses 'target_clones' for that purpose. A 'target("default")' attribute
// must therefore be treated as a plain codegen hint and must not create a
// multiversioned function, which would crash in
// CodeGenModule::emitMultiVersionFunctions.

int __attribute__((target("default"))) foo(void) { return 0; }
// CHECK: define {{.*}}i32 @foo()
