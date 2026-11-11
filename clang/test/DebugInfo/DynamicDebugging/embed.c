// REQUIRES: x86-registered-target
// Requires X86 as this test runs the codegen pipeline for the debug module.
// RUN: %clang -cc1 -emit-llvm -triple %itanium_abi_triple %s -debug-info-kind=limited -fdynamic-debugging -o - \
// RUN: | FileCheck %s

// Test that a dynamic debugging section is embedded in the outer module. Note
// that !exclude is ignored by LLVM as this section's flags are chosen based
// on its name. FIXME: We could introduce new metadata like !exclude to avoid
// the special casing in LLVM.
int e() { return 0; }

// CHECK: @llvm.embedded.object = private constant {{.*}}, section ".debug_llvm_dyndbg", align 8,
// CHECK-SAME: !elf_section_properties ![[elf_props:[0-9]+]], !metadata_section_kind
// CHECK: ![[elf_props:[0-9]+]] = !{i32 1879002128, i32 0}
