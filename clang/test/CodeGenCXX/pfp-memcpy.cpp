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
  // CHECK-NEXT: %call = call noundef nonnull align 8 dereferenceable(16) ptr @_ZN20ClassWithTrivialCopyaSERKS_(ptr noundef nonnull align 8 dereferenceable(16) %1, ptr noundef nonnull align 8 dereferenceable(16) %0)
  *s1 = *s2;
}

// CHECK-LABEL: define linkonce_odr noundef nonnull align 8 dereferenceable(16) ptr @_ZN20ClassWithTrivialCopyaSERKS_
// CHECK:      %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT: %1 = load ptr, ptr %.addr, align 8, !nonnull ![[#]], !align ![[#]]
// CHECK-NEXT: %a = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %1, i32 0, i32 0
// CHECK-NEXT: %2 = ptrtoint ptr %1 to i64
// CHECK-NEXT: %3 = call ptr @llvm.protected.field.ptr.p0(ptr %a, i64 %2, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.a) ]
// CHECK-NEXT: %4 = load ptr, ptr %3, align 8
// CHECK-NEXT: %a2 = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %this1, i32 0, i32 0
// CHECK-NEXT: %5 = ptrtoint ptr %this1 to i64
// CHECK-NEXT: %6 = call ptr @llvm.protected.field.ptr.p0(ptr %a2, i64 %5, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.a) ]
// CHECK-NEXT: store ptr %4, ptr %6, align 8
// CHECK-NEXT: %7 = load ptr, ptr %.addr, align 8, !nonnull ![[#]], !align ![[#]]
// CHECK-NEXT: %c = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %7, i32 0, i32 1
// CHECK-NEXT: %8 = ptrtoint ptr %7 to i64
// CHECK-NEXT: %9 = call ptr @llvm.protected.field.ptr.p0(ptr %c, i64 %8, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.c) ]
// CHECK-NEXT: %10 = load ptr, ptr %9, align 8
// CHECK-NEXT: %c3 = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %this1, i32 0, i32 1
// CHECK-NEXT: %11 = ptrtoint ptr %this1 to i64
// CHECK-NEXT: %12 = call ptr @llvm.protected.field.ptr.p0(ptr %c3, i64 %11, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.c) ]
// CHECK-NEXT: store ptr %10, ptr %12, align 8
// CHECK-NEXT: ret ptr %this1

void trivial_copy(ClassWithTrivialCopy *s1) {
  ClassWithTrivialCopy s2(*s1);
}

// CHECK-LABEL: define linkonce_odr void @_ZN20ClassWithTrivialCopyC2ERKS_
// CHECK:      %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT: %a = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %this1, i32 0, i32 0
// CHECK-NEXT: %1 = ptrtoint ptr %this1 to i64
// CHECK-NEXT: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %a, i64 %1, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.a) ]
// CHECK-NEXT: %3 = load ptr, ptr %.addr, align 8, !nonnull ![[#]], !align ![[#]]
// CHECK-NEXT: %a2 = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %3, i32 0, i32 0
// CHECK-NEXT: %4 = ptrtoint ptr %3 to i64
// CHECK-NEXT: %5 = call ptr @llvm.protected.field.ptr.p0(ptr %a2, i64 %4, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.a) ]
// CHECK-NEXT: %6 = load ptr, ptr %5, align 8
// CHECK-NEXT: store ptr %6, ptr %2, align 8
// CHECK-NEXT: %c = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %this1, i32 0, i32 1
// CHECK-NEXT: %7 = ptrtoint ptr %this1 to i64
// CHECK-NEXT: %8 = call ptr @llvm.protected.field.ptr.p0(ptr %c, i64 %7, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.c) ]
// CHECK-NEXT: %9 = load ptr, ptr %.addr, align 8, !nonnull ![[#]], !align ![[#]]
// CHECK-NEXT: %c3 = getelementptr inbounds nuw %struct.ClassWithTrivialCopy, ptr %9, i32 0, i32 1
// CHECK-NEXT: %10 = ptrtoint ptr %9 to i64
// CHECK-NEXT: %11 = call ptr @llvm.protected.field.ptr.p0(ptr %c3, i64 %10, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS20ClassWithTrivialCopy.c) ]
// CHECK-NEXT: %12 = load ptr, ptr %11, align 8
// CHECK-NEXT: store ptr %12, ptr %8, align 8
