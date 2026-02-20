// RUN: %clang_cc1 -triple aarch64-linux -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -emit-llvm -o - %s | FileCheck %s

struct ClassWithTrivialCopy {
  ClassWithTrivialCopy();
  ~ClassWithTrivialCopy();
  void *a;
private:
  void *c;
};

// Make sure that trivial assignments and copies include protected field copies.
// CHECK-LABEL: define dso_local void @_Z14trivial_assignP20ClassWithTrivialCopyS0_
void trivial_assign(ClassWithTrivialCopy *s1, ClassWithTrivialCopy *s2) {
  // CHECK:      %0 = load ptr, ptr %s2.addr, align 8
  // CHECK-NEXT: %1 = load ptr, ptr %s1.addr, align 8
  // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %1, ptr align 8 %0, i64 16, i1 false)
  // CHECK-NEXT: %2 = getelementptr inbounds i8, ptr %1, i64 0
  // CHECK-NEXT: %3 = ptrtoint ptr %1 to i64
  // CHECK-NEXT: %4 = call ptr @llvm.protected.field.ptr.p0(ptr %2, i64 %3, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.a) ]
  // CHECK-NEXT: %5 = getelementptr inbounds i8, ptr %0, i64 0
  // CHECK-NEXT: %6 = ptrtoint ptr %0 to i64
  // CHECK-NEXT: %7 = call ptr @llvm.protected.field.ptr.p0(ptr %5, i64 %6, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.a) ]
  // CHECK-NEXT: %8 = load ptr, ptr %7, align 8
  // CHECK-NEXT: store ptr %8, ptr %4, align 8
  // CHECK-NEXT: %9 = getelementptr inbounds i8, ptr %1, i64 8
  // CHECK-NEXT: %10 = ptrtoint ptr %1 to i64
  // CHECK-NEXT: %11 = call ptr @llvm.protected.field.ptr.p0(ptr %9, i64 %10, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.c) ]
  // CHECK-NEXT: %12 = getelementptr inbounds i8, ptr %0, i64 8
  // CHECK-NEXT: %13 = ptrtoint ptr %0 to i64
  // CHECK-NEXT: %14 = call ptr @llvm.protected.field.ptr.p0(ptr %12, i64 %13, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.c) ]
  // CHECK-NEXT: %15 = load ptr, ptr %14, align 8
  // CHECK-NEXT: store ptr %15, ptr %11, align 8
  *s1 = *s2;
}

void trivial_copy(ClassWithTrivialCopy *s1) {
  ClassWithTrivialCopy s2(*s1);
}

// CHECK-LABEL: define linkonce_odr void @_ZN20ClassWithTrivialCopyC2ERKS_
// CHECK:      %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT: %a = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %this1, i32 0, i32 0
// CHECK-NEXT: %1 = ptrtoint ptr %this1 to i64
// CHECK-NEXT: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %a, i64 %1, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.a) ]
// CHECK-NEXT: %3 = load ptr, ptr %.addr, align 8, !nonnull !4, !align !5
// CHECK-NEXT: %a2 = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %3, i32 0, i32 0
// CHECK-NEXT: %4 = ptrtoint ptr %3 to i64
// CHECK-NEXT: %5 = call ptr @llvm.protected.field.ptr.p0(ptr %a2, i64 %4, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.a) ]
// CHECK-NEXT: %6 = load ptr, ptr %5, align 8
// CHECK-NEXT: store ptr %6, ptr %2, align 8
// CHECK-NEXT: %c = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %this1, i32 0, i32 1
// CHECK-NEXT: %7 = ptrtoint ptr %this1 to i64
// CHECK-NEXT: %8 = call ptr @llvm.protected.field.ptr.p0(ptr %c, i64 %7, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.c) ]
// CHECK-NEXT: %9 = load ptr, ptr %.addr, align 8, !nonnull !4, !align !5
// CHECK-NEXT: %c3 = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %9, i32 0, i32 1
// CHECK-NEXT: %10 = ptrtoint ptr %9 to i64
// CHECK-NEXT: %11 = call ptr @llvm.protected.field.ptr.p0(ptr %c3, i64 %10, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.c) ]
// CHECK-NEXT: %12 = load ptr, ptr %11, align 8
// CHECK-NEXT: store ptr %12, ptr %8, align 8
