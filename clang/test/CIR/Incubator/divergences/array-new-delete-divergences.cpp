// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Array new/delete divergences:
// 1. Missing nobuiltin attribute on operator new[] and operator delete[]
// 2. Missing allocsize(0) attribute on operator new[]
// 3. Missing null check before delete[]
// 4. Missing inbounds on getelementptr
// 5. Missing noundef/nonnull on function declarations
//
// CodeGen:
//   declare noundef nonnull ptr @_Znam(i64 noundef) #1
//   ; Function Attrs: nobuiltin allocsize(0)
//
//   declare void @_ZdaPv(ptr noundef) #2
//   ; Function Attrs: nobuiltin nounwind
//
//   %isnull = icmp eq ptr %arr, null
//   br i1 %isnull, label %delete.end, label %delete.notnull
//   delete.notnull:
//     call void @_ZdaPv(ptr noundef %arr)
//
//   %arrayidx = getelementptr inbounds i32, ptr %arr, i64 0
//
// CIR:
//   declare ptr @_Znam(i64)  (missing noundef, nonnull, nobuiltin, allocsize)
//   declare void @_ZdaPv(ptr)  (missing noundef, nobuiltin, nounwind)
//
//   call void @_ZdaPv(ptr %arr)  (no null check)
//
//   %arrayidx = getelementptr i32, ptr %arr, i64 0  (missing inbounds)

// DIFF: -declare noundef nonnull ptr @_Znam(i64 noundef)
// DIFF: +declare ptr @_Znam(i64)
// DIFF: -; Function Attrs: nobuiltin allocsize(0)
// DIFF: -declare void @_ZdaPv(ptr noundef)
// DIFF: +declare void @_ZdaPv(ptr)
// DIFF: -; Function Attrs: nobuiltin nounwind
// DIFF: -%isnull = icmp eq ptr
// DIFF: -br i1 %isnull
// DIFF: -delete.notnull:
// DIFF: getelementptr inbounds
// DIFF: -getelementptr inbounds
// DIFF: +getelementptr

int test() {
    int* arr = new int[10];
    arr[0] = 42;
    int result = arr[0];
    delete[] arr;
    return result;
}
