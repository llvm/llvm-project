// REQUIRES: x86-registered-target

// This verifies that we correctly handle a -fms-secure-hotpatch-functions-file argument that points
// to a missing file.
//
// RUN: not %clang_cl -c --target=x86_64-windows-msvc -O2 /Z7 -fms-secure-hotpatch-functions-file=%S/this-file-is-intentionally-missing-do-not-create-it.txt /Fo%t.obj -- %s 2>&1 | FileCheck %s
// CHECK: failed to open hotpatch functions file

void this_might_have_side_effects();

int __declspec(noinline) this_gets_hotpatched() {
    this_might_have_side_effects();
    return 42;
}

int __declspec(noinline) this_does_not_get_hotpatched() {
    return this_gets_hotpatched() + 100;
}
