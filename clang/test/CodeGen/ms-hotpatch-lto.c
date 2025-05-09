// This verifies that hotpatch function attributes are correctly propagated through LLVM IR when compiling with LTO.
//
// RUN: echo this_gets_hotpatched > %t-hotpatch-functions.txt
// RUN: %clang_cl -c --target=x86_64-windows-msvc -O2 /Z7 -fms-hotpatch-functions-file=%t-hotpatch-functions.txt -flto /Fo%t.bc %s
// RUN: llvm-dis %t.bc -o - | FileCheck %s
//
// CHECK: ; Function Attrs: marked_for_windows_hot_patching mustprogress nofree noinline norecurse nosync nounwind sspstrong willreturn memory(none) uwtable
// CHECK-NEXT: define dso_local noundef i32 @this_gets_hotpatched()
//
// CHECK: ; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind sspstrong willreturn memory(none) uwtable
// CHECK-NEXT: define dso_local noundef i32 @this_does_not_get_hotpatched()

int __declspec(noinline) this_gets_hotpatched() {
    return 42;
}

int __declspec(noinline) this_does_not_get_hotpatched() {
    return this_gets_hotpatched() + 100;
}
