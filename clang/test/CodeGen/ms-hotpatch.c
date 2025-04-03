// This verifies that hotpatch function attributes are correctly propagated when compiling directly to OBJ.
//
// RUN: %clang_cl -c --target=x86_64-windows-msvc -O2 /Z7 -fms-hotpatch-functions-file=%S/ms-hotpatch-functions.txt /Fo%t.obj %s
// RUN: llvm-readobj --codeview %t.obj | FileCheck %s
// CHECK: Kind: S_HOTPATCHFUNC (0x1169)
// CHECK-NEXT: Function: this_gets_hotpatched

void this_might_have_side_effects();

int __declspec(noinline) this_gets_hotpatched() {
    this_might_have_side_effects();
    return 42;
}

int __declspec(noinline) this_does_not_get_hotpatched() {
    return this_gets_hotpatched() + 100;
}
