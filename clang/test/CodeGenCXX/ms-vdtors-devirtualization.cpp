// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - | FileCheck --check-prefixes=CHECK,X64 %s
// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=i386-pc-windows-msvc -o - | FileCheck --check-prefixes=CHECK,X86 %s

struct Base {
  virtual ~Base() {}
};

struct A final :  Base {
  virtual ~A();
};

struct B : Base { virtual ~B() final {} };

struct D { virtual ~D() final = 0; };

void case1(A *arg) {
  delete[] arg;
}
// X64-LABEL: define {{.*}} void @"?case1@@YAXPEAUA@@@Z"
// X64-SAME: (ptr noundef %[[ARG:.*]])
// X86-LABEL: define {{.*}} void @"?case1@@YAXPAUA@@@Z"
// X86-SAME: (ptr noundef %[[ARG:.*]])
// CHECK: entry:
// CHECK-NEXT:  %[[ARGADDR:.*]] = alloca ptr
// CHECK-NEXT:  store ptr %[[ARG]], ptr %[[ARGADDR]],
// CHECK-NEXT:  %[[ARR:.*]] = load ptr, ptr %[[ARGADDR]]
// CHECK-NEXT:  %[[ISNULL:.*]] = icmp eq ptr %[[ARR]], null
// CHECK-NEXT:  br i1 %[[ISNULL]], label %delete.end2, label %delete.notnull
// CHECK:  delete.notnull:
// X64-NEXT:  %[[COOKIEADDR:.*]] = getelementptr inbounds i8, ptr %[[ARR]], i64 -8
// X86-NEXT:  %[[COOKIEADDR:.*]] = getelementptr inbounds i8, ptr %0, i32 -4
// X64-NEXT:  %[[COOKIE:.*]] = load i64, ptr %[[COOKIEADDR]]
// X86-NEXT:  %[[COOKIE:.*]] = load i32, ptr %[[COOKIEADDR]]
// X64-NEXT:  %[[END:.*]] = getelementptr inbounds %struct.A, ptr %[[ARR]], i64 %[[COOKIE]]
// X86-NEXT:  %[[END:.*]] = getelementptr inbounds %struct.A, ptr %[[ARR]], i32 %[[COOKIE]]
// CHECK-NEXT:  %[[ISEMPTY:.*]] = icmp eq ptr %[[ARR]], %[[END]]
// CHECK-NEXT:  br i1 %arraydestroy.isempty, label %arraydestroy.done1, label %arraydestroy.body
// CHECK: arraydestroy.body:
// CHECK-NEXT:  %arraydestroy.elementPast = phi ptr [ %delete.end, %delete.notnull ], [ %arraydestroy.element, %arraydestroy.body ]
// X64-NEXT:  %arraydestroy.element = getelementptr inbounds %struct.A, ptr %arraydestroy.elementPast, i64 -1
// X86-NEXT:  %arraydestroy.element = getelementptr inbounds %struct.A, ptr %arraydestroy.elementPast, i32 -1
// X64-NEXT:  call void @"??1A@@UEAA@XZ"(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %arraydestroy.element)
// X86-NEXT:  call x86_thiscallcc void @"??1A@@UAE@XZ"(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %arraydestroy.element)
// CHECK-NEXT:  %arraydestroy.done = icmp eq ptr %arraydestroy.element, %0
// CHECK-NEXT:  br i1 %arraydestroy.done, label %arraydestroy.done1, label %arraydestroy.body
// CHECK:  arraydestroy.done1:
// X64-NEXT:  %[[HOWMANYELEMS:.*]] = mul i64 8, %[[COOKIE]]
// X86-NEXT:  %[[HOWMANYELEMS:.*]] = mul i32 4, %[[COOKIE]]
// X64-NEXT:  %[[SIZEPLUSCOOKIE:.*]] = add i64 %[[HOWMANYELEMS]], 8
// X86-NEXT:  %[[SIZEPLUSCOOKIE:.*]] = add i32 %[[HOWMANYELEMS]], 4
// X64-NEXT:  call void @"??_V@YAXPEAX_K@Z"(ptr noundef %[[COOKIEADDR]], i64 noundef %[[SIZEPLUSCOOKIE]])
// X86-NEXT:  call void @"??_V@YAXPAXI@Z"(ptr noundef %[[COOKIEADDR]], i32 noundef %[[SIZEPLUSCOOKIE]])
// CHECK-NEXT:  br label %delete.end2

void case2(B *arg) {
  delete[] arg;
}

// X64-LABEL: define {{.*}} void @"?case2@@YAXPEAUB@@@Z"
// X64-SAME: (ptr noundef %[[ARG:.*]])
// X86-LABEL: define {{.*}} void @"?case2@@YAXPAUB@@@Z"
// X86-SAME: (ptr noundef %[[ARG:.*]])
// CHECK: entry:
// CHECK-NEXT:  %[[ARGADDR:.*]] = alloca ptr
// CHECK-NEXT:  store ptr %[[ARG]], ptr %[[ARGADDR]],
// CHECK-NEXT:  %[[ARR:.*]] = load ptr, ptr %[[ARGADDR]]
// CHECK-NEXT:  %[[ISNULL:.*]] = icmp eq ptr %[[ARR]], null
// CHECK-NEXT:  br i1 %[[ISNULL]], label %delete.end2, label %delete.notnull
// CHECK:  delete.notnull:
// X64-NEXT:  %[[COOKIEADDR:.*]] = getelementptr inbounds i8, ptr %[[ARR]], i64 -8
// X86-NEXT:  %[[COOKIEADDR:.*]] = getelementptr inbounds i8, ptr %0, i32 -4
// X64-NEXT:  %[[COOKIE:.*]] = load i64, ptr %[[COOKIEADDR]]
// X86-NEXT:  %[[COOKIE:.*]] = load i32, ptr %[[COOKIEADDR]]
// X64-NEXT:  %[[END:.*]] = getelementptr inbounds %struct.B, ptr %[[ARR]], i64 %[[COOKIE]]
// X86-NEXT:  %[[END:.*]] = getelementptr inbounds %struct.B, ptr %[[ARR]], i32 %[[COOKIE]]
// CHECK-NEXT:  %[[ISEMPTY:.*]] = icmp eq ptr %[[ARR]], %[[END]]
// CHECK-NEXT:  br i1 %arraydestroy.isempty, label %arraydestroy.done1, label %arraydestroy.body
// CHECK: arraydestroy.body:
// CHECK-NEXT:  %arraydestroy.elementPast = phi ptr [ %delete.end, %delete.notnull ], [ %arraydestroy.element, %arraydestroy.body ]
// X64-NEXT:  %arraydestroy.element = getelementptr inbounds %struct.B, ptr %arraydestroy.elementPast, i64 -1
// X86-NEXT:  %arraydestroy.element = getelementptr inbounds %struct.B, ptr %arraydestroy.elementPast, i32 -1
// X64-NEXT:  call void @"??1B@@UEAA@XZ"(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %arraydestroy.element)
// X86-NEXT:  call x86_thiscallcc void @"??1B@@UAE@XZ"(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %arraydestroy.element)
// CHECK-NEXT:  %arraydestroy.done = icmp eq ptr %arraydestroy.element, %0
// CHECK-NEXT:  br i1 %arraydestroy.done, label %arraydestroy.done1, label %arraydestroy.body
// CHECK:  arraydestroy.done1:
// X64-NEXT:  %[[HOWMANYELEMS:.*]] = mul i64 8, %[[COOKIE]]
// X86-NEXT:  %[[HOWMANYELEMS:.*]] = mul i32 4, %[[COOKIE]]
// X64-NEXT:  %[[SIZEPLUSCOOKIE:.*]] = add i64 %[[HOWMANYELEMS]], 8
// X86-NEXT:  %[[SIZEPLUSCOOKIE:.*]] = add i32 %[[HOWMANYELEMS]], 4
// X64-NEXT:  call void @"??_V@YAXPEAX_K@Z"(ptr noundef %[[COOKIEADDR]], i64 noundef %[[SIZEPLUSCOOKIE]])
// X86-NEXT:  call void @"??_V@YAXPAXI@Z"(ptr noundef %[[COOKIEADDR]], i32 noundef %[[SIZEPLUSCOOKIE]])
// CHECK-NEXT:  br label %delete.end2


void case3(D *arg) {
  delete[] arg;
}

// CHECK-LABEL: case3
// X64: call noundef ptr %{{.}}(
// X86: call x86_thiscallcc noundef ptr %{{.}}(

void case4(D **arg) {
  delete[] arg[0];
  delete[] arg[1];
}

// CHECK-LABEL: case4
// X64: call noundef ptr %{{.}}(
// X86: call x86_thiscallcc noundef ptr %{{.}}(
