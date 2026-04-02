// REQUIRES: x86-registered-target
// Requires X86 as this test runs the codegen pipeline for the debug module.
// RUN: %clang -cc1 -triple %itanium_abi_triple %s -debug-info-kind=limited -fdynamic-debugging -o %t --save-dynamic-debugging-temps
// RUN: FileCheck %s < %t.dyndbg.2.outer.ll

// Test that a dynamic debugging section is embedded in the outer module. Note
// that !exclude is ignored by LLVM as this section's flags are chosen based
// on its name. FIXME: We could introduce new metadata like !exclude to avoid
// the special casing in LLVM.
int e() { return 0; }

// CHECK: @llvm.embedded.object = private constant {{.*}}, section ".debug_llvm_dyndbg", align 1, !exclude
