// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - | FileCheck --check-prefixes=X64,CHECK %s
// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=i386-pc-windows-msvc -o - | FileCheck --check-prefixes=X86,CHECK %s

using size_t = __SIZE_TYPE__;
void operator delete[](void *ptr, size_t sz) { }

struct Test {
  virtual ~Test() {}
  void operator delete[](void *ptr, size_t sz) {  }
  int x;
  int y;
};

void test() {
  Test* a = new Test[10];
  delete[] a;
}

// X64: define weak dso_local noundef ptr @"??_ETest@@UEAAPEAXI@Z"(
// X64-SAME: ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[IMPLICIT_PARAM:.*]])
// X86: define weak dso_local x86_thiscallcc noundef ptr @"??_ETest@@UAEPAXI@Z"(
// X86-SAME: ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[IMPLICIT_PARAM:.*]])
// CHECK: entry:
// CHECK-NEXT:   %[[RET:.*]] = alloca ptr
// CHECK-NEXT:   %[[IPADDR:.*]] = alloca i32
// CHECK-NEXT:   %[[THISADDR:.*]] = alloca ptr
// CHECK-NEXT:   store i32 %[[IMPLICIT_PARAM]], ptr %[[IPADDR]]
// CHECK-NEXT:   store ptr %[[THIS]], ptr %[[THISADDR]]
// CHECK-NEXT:   %[[LTHIS:.*]] = load ptr, ptr %[[THISADDR]]
// CHECK-NEXT:   store ptr %[[LTHIS]], ptr %[[RET]]
// CHECK-NEXT:   %[[LIP:.*]] = load i32, ptr %[[IPADDR]]
// CHECK-NEXT:   %[[SECONDBIT:.*]] = and i32 %[[LIP]], 2
// CHECK-NEXT:   %[[ISSECONDBITZERO:.*]] = icmp eq i32 %[[SECONDBIT]], 0
// CHECK-NEXT:   br i1 %[[ISSECONDBITZERO:.*]], label %dtor.scalar, label %dtor.vector
// CHECK: dtor.vector:
// X64-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[LTHIS]], i64 -8
// X86-NEXT:   %[[COOKIEGEP:.*]] = getelementptr inbounds i8, ptr %[[LTHIS]], i32 -4
// X64-NEXT:   %[[HOWMANY:.*]] = load i64, ptr %[[COOKIEGEP]]
// X86-NEXT:   %[[HOWMANY:.*]] = load i32, ptr %[[COOKIEGEP]]
// CHECK: dtor.call_class_delete_after_array_destroy:
// X64-NEXT:  %[[ARRSZ:.*]] = mul i64 16, %[[HOWMANY]]
// X86-NEXT:  %[[ARRSZ:.*]] = mul i32 12, %[[HOWMANY]]
// X64-NEXT:  %[[TOTALSZ:.*]] = add i64 %[[ARRSZ]], 8
// X86-NEXT:  %[[TOTALSZ:.*]] = add i32 %[[ARRSZ]], 4
// X64-NEXT:   call void @"??_VTest@@SAXPEAX_K@Z"(ptr noundef %2, i64 noundef %[[TOTALSZ]])
// X86-NEXT:   call void @"??_VTest@@SAXPAXI@Z"(ptr noundef %2, i32 noundef %[[TOTALSZ]])

// CHECK: dtor.call_glob_delete_after_array_destroy:
// X64-NEXT:  %[[ARRSZ1:.*]] = mul i64 16, %[[HOWMANY]]
// X86-NEXT:  %[[ARRSZ1:.*]] = mul i32 12, %[[HOWMANY]]
// X64-NEXT:  %[[TOTALSZ1:.*]] = add i64 %[[ARRSZ1]], 8
// X86-NEXT:  %[[TOTALSZ1:.*]] = add i32 %[[ARRSZ1]], 4
// X64-NEXT:   call void @"??_V@YAXPEAX_K@Z"(ptr noundef %2, i64 noundef %[[TOTALSZ1]])
// X86-NEXT:   call void @"??_V@YAXPAXI@Z"(ptr noundef %2, i32 noundef %[[TOTALSZ1]])
