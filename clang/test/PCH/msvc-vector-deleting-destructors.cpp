// Test this without pch.
// RUN: %clang_cc1 -x c++ -include %S/Inputs/msvc-vector-deleting-dtors.h -emit-llvm -o - %s -triple=i386-pc-win32 | FileCheck %s --check-prefixes CHECK,CHECK32
// RUN: %clang_cc1 -x c++ -include %S/Inputs/msvc-vector-deleting-dtors.h -emit-llvm -o - %s -triple=x86_64-pc-win32 | FileCheck %s --check-prefixes CHECK,CHECK64

// Test with pch.
// RUN: %clang_cc1 -x c++ -emit-pch -o %t -triple=i386-pc-win32 %S/Inputs/msvc-vector-deleting-dtors.h
// RUN: %clang_cc1 -x c++ -include-pch %t -emit-llvm -triple=i386-pc-win32 -o - %s | FileCheck %s --check-prefixes CHECK,CHECK32
// RUN: %clang_cc1 -x c++ -emit-pch -o %t -triple=x86_64-pc-win32 %S/Inputs/msvc-vector-deleting-dtors.h
// RUN: %clang_cc1 -x c++ -include-pch %t -emit-llvm -triple=x86_64-pc-win32 -o - %s | FileCheck %s --check-prefixes CHECK,CHECK64

void call_in_module_function(void) {
    in_h_tests(new Derived[2], new Derived[3]);
}

void out_of_module_tests(Derived *p, Derived *p1) {
  ::delete[] p;

  delete[] p1;
}

// CHECK32-LABEL: define weak dso_local x86_thiscallcc noundef ptr @"??_EDerived@@UAEPAXI@Z"
// CHECK64-LABEL: define weak dso_local noundef ptr @"??_EDerived@@UEAAPEAXI@Z"
// CHECK: dtor.call_class_delete_after_array_destroy:
// CHECK32-NEXT:  call void @"??_VBase1@@SAXPAX@Z"(ptr noundef %2)
// CHECK64-NEXT:  call void @"??_VBase1@@SAXPEAX@Z"(ptr noundef %2)
// CHECK: dtor.call_glob_delete_after_array_destroy:
// CHECK32-NEXT:   call void @"??_V@YAXPAXI@Z"(ptr noundef %2, i32 noundef 8)
// CHECK64-NEXT:   call void @"??_V@YAXPEAX_K@Z"(ptr noundef %2, i64 noundef 16)
// CHECK: dtor.call_glob_delete:
// CHECK32-NEXT:   call void @"??3@YAXPAXI@Z"(ptr noundef %this1, i32 noundef 8)
// CHECK64-NEXT:   call void @"??3@YAXPEAX_K@Z"(ptr noundef %this1, i64 noundef 16)
// CHECK: dtor.call_class_delete:
// CHECK32-NEXT:   call void @"??3Base2@@SAXPAX@Z"(ptr noundef %this1)
// CHECK64-NEXT:   call void @"??3Base2@@SAXPEAX@Z"(ptr noundef %this1)
