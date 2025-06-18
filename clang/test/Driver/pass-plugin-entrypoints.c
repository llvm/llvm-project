// REQUIRES: pass-plugins
// UNSUPPORTED: system-windows

// Test default entry-point
// RUN: %clang -O0 -fpass-plugin=%pass_plugin_reference \
// RUN:        -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | \
// RUN:        FileCheck --check-prefix=EP-EARLY-SIMPL %s
//
// RUN: %clang -O2 -fpass-plugin=%pass_plugin_reference \
// RUN:        -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | \
// RUN:        FileCheck --check-prefixes=EP-EARLY-SIMPL,LTO-NONE %s
//
// RUN: %clang -c -flto=full -O2 -fpass-plugin=%pass_plugin_reference \
// RUN:        -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | \
// RUN:        FileCheck --check-prefixes=EP-EARLY-SIMPL,LTO-FULL %s
//
// RUN: %clang -c -flto=thin -O2 -fpass-plugin=%pass_plugin_reference \
// RUN:        -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | \
// RUN:        FileCheck --check-prefixes=EP-EARLY-SIMPL,LTO-THIN %s
//
// EP-EARLY-SIMPL: Running pass: TestModulePass
// EP-EARLY-SIMPL: Entry-point: registerPipelineEarlySimplificationEPCallback
//
// LTO-NONE: LTO-phase: None
// LTO-FULL: LTO-phase: FullLTOPreLink
// LTO-THIN: LTO-phase: ThinLTOPreLink

// Pass doesn't run if default entry-point is disabled
// RUN: env registerPipelineEarlySimplificationEPCallback=Off \
// RUN:     %clang -fpass-plugin=%pass_plugin_reference -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck %s
//
// CHECK-NOT: Running pass: TestModulePass

// Test pipeline-start entry-point
// RUN: env registerPipelineStartEPCallback=On \
// RUN: env registerPipelineEarlySimplificationEPCallback=Off \
// RUN:     %clang -fpass-plugin=%pass_plugin_reference -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck --check-prefix=EP-START %s
//
// EP-START: Running pass: TestModulePass
// EP-START: Entry-point: registerPipelineStartEPCallback

// Test optimizer entry-points
// RUN: env registerOptimizerEarlyEPCallback=On \
// RUN: env registerPipelineEarlySimplificationEPCallback=Off \
// RUN:     %clang -fpass-plugin=%pass_plugin_reference -O2 -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck --check-prefix=OPT-EARLY %s
//
// OPT-EARLY: Running pass: TestModulePass
// OPT-EARLY: Entry-point: registerOptimizerEarlyEPCallback
// OPT-EARLY: Running pass: LowerConstantIntrinsicsPass

#if LLVM_VERSION_MAJOR > 20
// RUN: env registerOptimizerLastEPCallback=On \
// RUN: env registerPipelineEarlySimplificationEPCallback=Off \
// RUN:     %clang -fpass-plugin=%pass_plugin_reference -O2 -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck --check-prefix=OPT-LAST %s
//
// OPT-LAST: Running pass: LowerConstantIntrinsicsPass
// OPT-LAST: Running pass: TestModulePass on [module]
// OPT-LAST: Entry-point: registerOptimizerLastEPCallback
#endif

// TODO: Why is late optimizer entry-point reached in LTO-mode??
// RUN: env registerOptimizerLastEPCallback=On \
// RUN: env registerPipelineEarlySimplificationEPCallback=Off \
// RUN:     %clang -fpass-plugin=%pass_plugin_reference -O2 -flto -c -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1

int main() { return 0; }
