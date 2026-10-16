// REQUIRES: x86-registered-target

// This verifies that hotpatch function attributes are correctly propagated through LLVM IR when compiling with LTO.
//
// RUN: %clang_cl -c --target=x86_64-windows-msvc -O2 /Z7 -fms-secure-hotpatch-functions-list=this_gets_hotpatched -flto /Fo%t.bc -- %s
// RUN: llvm-dis %t.bc -o - | FileCheck %s
//
// CHECK-LABEL: define dso_local noundef i32 @this_gets_hotpatched()
// CHECK-SAME: #0
//
// CHECK-LABEL: define dso_local noundef i32 @this_does_not_get_hotpatched()
// CHECK-SAME: #1

// CHECK: attributes #0
// CHECK-SAME: "marked_for_windows_hot_patching"

// CHECK: attributes #1
// CHECK-NOT: "marked_for_windows_hot_patching"

int __declspec(noinline) this_gets_hotpatched() {
    return 42;
}

int __declspec(noinline) this_does_not_get_hotpatched() {
    return this_gets_hotpatched() + 100;
}
