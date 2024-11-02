// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s
// Verify that the contextual-{final,override} rules are guarded conditionally,
// No ambiguous parsing for the virt-specifier.
class Foo {
    void foo1() override;
// CHECK: virt-specifier-seq~IDENTIFIER := tok[7]
    void foo2() final;
// CHECK: virt-specifier-seq~IDENTIFIER := tok[13]
};
