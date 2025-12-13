// RUN: %clang_cc1 -std=c++26 -triple x86_64-linux-gnu -emit-llvm -fexperimental-pointer-field-protection-abi -o - %s | FileCheck --check-prefix=RELOC %s
// RUN: %clang_cc1 -std=c++26 -triple x86_64-linux-gnu -emit-llvm -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -o - %s | FileCheck --check-prefix=RELOC %s
// RUN: %clang_cc1 -std=c++26 -triple aarch64-linux-gnu -emit-llvm -fexperimental-pointer-field-protection-abi -o - %s | FileCheck --check-prefix=RELOC %s
// RUN: %clang_cc1 -std=c++26 -triple aarch64-linux-gnu -emit-llvm -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -o - %s | FileCheck --check-prefix=NONRELOC %s

typedef __SIZE_TYPE__ size_t;

struct S trivially_relocatable_if_eligible {
    S(const S&);
    ~S();
    int* a;
private:
    int* b;
};

// CHECK: define dso_local void @_Z5test1P1SS0_(
void test1(S* source, S* dest) {
  // RELOC:       %0 = load ptr, ptr %dest.addr, align 8
  // RELOC-NEXT:  %1 = load ptr, ptr %source.addr, align 8
  // RELOC-NEXT:  call void @llvm.memmove.p0.p0.i64(ptr align 8 %0, ptr align 8 %1, i64 16, i1 false)
  // RELOC-NOT: @llvm.protected.field.ptr.p0

  // NONRELOC:        %0 = load ptr, ptr %dest.addr, align 8
  // NONRELOC-NEXT:   %1 = load ptr, ptr %source.addr, align 8
  // NONRELOC-NEXT:   call void @llvm.memmove.p0.p0.i64(ptr align 8 %0, ptr align 8 %1, i64 16, i1 false)
  // NONRELOC-NEXT:   br i1 false, label %pfp.relocate.loop.end, label %pfp.relocate.loop

  // NONRELOC:      pfp.relocate.loop:
  // NONRELOC-NEXT:   %2 = phi i64 [ 0, %entry ], [ %19, %pfp.relocate.loop ]
  // NONRELOC-NEXT:   %3 = getelementptr inbounds i8, ptr %0, i64 %2
  // NONRELOC-NEXT:   %4 = getelementptr inbounds i8, ptr %1, i64 %2
  // NONRELOC-NEXT:   %5 = getelementptr inbounds i8, ptr %3, i64 0
  // NONRELOC-NEXT:   %6 = ptrtoint ptr %3 to i64
  // NONRELOC-NEXT:   %7 = call ptr @llvm.protected.field.ptr.p0(ptr %5, i64 %6, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.a) ]
  // NONRELOC-NEXT:   %8 = getelementptr inbounds i8, ptr %4, i64 0
  // NONRELOC-NEXT:   %9 = ptrtoint ptr %4 to i64
  // NONRELOC-NEXT:   %10 = call ptr @llvm.protected.field.ptr.p0(ptr %8, i64 %9, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.a) ]
  // NONRELOC-NEXT:   %11 = load ptr, ptr %10, align 8
  // NONRELOC-NEXT:   store ptr %11, ptr %7, align 8
  // NONRELOC-NEXT:   %12 = getelementptr inbounds i8, ptr %3, i64 8
  // NONRELOC-NEXT:   %13 = ptrtoint ptr %3 to i64
  // NONRELOC-NEXT:   %14 = call ptr @llvm.protected.field.ptr.p0(ptr %12, i64 %13, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.b) ]
  // NONRELOC-NEXT:   %15 = getelementptr inbounds i8, ptr %4, i64 8
  // NONRELOC-NEXT:   %16 = ptrtoint ptr %4 to i64
  // NONRELOC-NEXT:   %17 = call ptr @llvm.protected.field.ptr.p0(ptr %15, i64 %16, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.b) ]
  // NONRELOC-NEXT:   %18 = load ptr, ptr %17, align 8
  // NONRELOC-NEXT:   store ptr %18, ptr %14, align 8
  // NONRELOC-NEXT:   %19 = add i64 %2, 16
  // NONRELOC-NEXT:   %20 = icmp eq i64 %19, 16
  // NONRELOC-NEXT:   br i1 %20, label %pfp.relocate.loop.end, label %pfp.relocate.loop

  // NONRELOC:      pfp.relocate.loop.end:
  // NONRELOC-NEXT:   ret void
  __builtin_trivially_relocate(dest, source, 1);
}

// CHECK: define dso_local void @_Z5testNP1SS0_m(
void testN(S* source, S* dest, size_t count) {
  // RELOC:       %0 = load ptr, ptr %dest.addr, align 8
  // RELOC-NEXT:  %1 = load ptr, ptr %source.addr, align 8
  // RELOC-NEXT:  %2 = load i64, ptr %count.addr, align 8
  // RELOC-NEXT:  %3 = mul i64 %2, 16
  // RELOC-NEXT:  call void @llvm.memmove.p0.p0.i64(ptr align 8 %0, ptr align 8 %1, i64 %3, i1 false)
  // RELOC-NOT: @llvm.protected.field.ptr.p0
 
  // NONRELOC:        %0 = load ptr, ptr %dest.addr, align 8
  // NONRELOC-NEXT:   %1 = load ptr, ptr %source.addr, align 8
  // NONRELOC-NEXT:   %2 = load i64, ptr %count.addr, align 8
  // NONRELOC-NEXT:   %3 = mul i64 %2, 16
  // NONRELOC-NEXT:   call void @llvm.memmove.p0.p0.i64(ptr align 8 %0, ptr align 8 %1, i64 %3, i1 false)
  // NONRELOC-NEXT:   %4 = icmp eq i64 %3, 0
  // NONRELOC-NEXT:   br i1 %4, label %pfp.relocate.loop.end, label %pfp.relocate.loop

  // NONRELOC:      pfp.relocate.loop:
  // NONRELOC-NEXT:   %5 = phi i64 [ 0, %entry ], [ %22, %pfp.relocate.loop ]
  // NONRELOC-NEXT:   %6 = getelementptr inbounds i8, ptr %0, i64 %5
  // NONRELOC-NEXT:   %7 = getelementptr inbounds i8, ptr %1, i64 %5
  // NONRELOC-NEXT:   %8 = getelementptr inbounds i8, ptr %6, i64 0
  // NONRELOC-NEXT:   %9 = ptrtoint ptr %6 to i64
  // NONRELOC-NEXT:   %10 = call ptr @llvm.protected.field.ptr.p0(ptr %8, i64 %9, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.a) ]
  // NONRELOC-NEXT:   %11 = getelementptr inbounds i8, ptr %7, i64 0
  // NONRELOC-NEXT:   %12 = ptrtoint ptr %7 to i64
  // NONRELOC-NEXT:   %13 = call ptr @llvm.protected.field.ptr.p0(ptr %11, i64 %12, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.a) ]
  // NONRELOC-NEXT:   %14 = load ptr, ptr %13, align 8
  // NONRELOC-NEXT:   store ptr %14, ptr %10, align 8
  // NONRELOC-NEXT:   %15 = getelementptr inbounds i8, ptr %6, i64 8
  // NONRELOC-NEXT:   %16 = ptrtoint ptr %6 to i64
  // NONRELOC-NEXT:   %17 = call ptr @llvm.protected.field.ptr.p0(ptr %15, i64 %16, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.b) ]
  // NONRELOC-NEXT:   %18 = getelementptr inbounds i8, ptr %7, i64 8
  // NONRELOC-NEXT:   %19 = ptrtoint ptr %7 to i64
  // NONRELOC-NEXT:   %20 = call ptr @llvm.protected.field.ptr.p0(ptr %18, i64 %19, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.b) ]
  // NONRELOC-NEXT:   %21 = load ptr, ptr %20, align 8
  // NONRELOC-NEXT:   store ptr %21, ptr %17, align 8
  // NONRELOC-NEXT:   %22 = add i64 %5, 16
  // NONRELOC-NEXT:   %23 = icmp eq i64 %22, %3
  // NONRELOC-NEXT:   br i1 %23, label %pfp.relocate.loop.end, label %pfp.relocate.loop

  // NONRELOC:      pfp.relocate.loop.end:
  // NONRELOC-NEXT:   ret void
  __builtin_trivially_relocate(dest, source, count);
};
