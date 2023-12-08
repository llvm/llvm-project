// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s --check-prefix=CHECK-32

extern unsigned long a;
extern const void *b;
extern void *c;

// CHECK-LABEL: @test_popcntb(
// CHECK:    [[TMP0:%.*]] = load i64, ptr @a, align 8
// CHECK-NEXT:    [[POPCNTB:%.*]] = call i64 @llvm.ppc.popcntb.i64.i64(i64 [[TMP0]])
// CHECK-NEXT:    ret i64 [[POPCNTB]]
//
// CHECK-32-LABEL: @test_popcntb(
// CHECK-32:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-32-NEXT:    [[POPCNTB:%.*]] = call i32 @llvm.ppc.popcntb.i32.i32(i32 [[TMP0]])
// CHECK-32-NEXT:    ret i32 [[POPCNTB]]
//
unsigned long test_popcntb() {
  return __popcntb(a);
}

// CHECK-LABEL: @test_eieio(
// CHECK:    call void @llvm.ppc.eieio()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_eieio(
// CHECK-32:    call void @llvm.ppc.eieio()
// CHECK-32-NEXT:    ret void
//
void test_eieio() {
  __eieio();
}

// CHECK-LABEL: @test_iospace_eieio(
// CHECK:    call void @llvm.ppc.iospace.eieio()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_iospace_eieio(
// CHECK-32:    call void @llvm.ppc.iospace.eieio()
// CHECK-32-NEXT:    ret void
//
void test_iospace_eieio() {
  __iospace_eieio();
}

// CHECK-LABEL: @test_isync(
// CHECK:    call void @llvm.ppc.isync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_isync(
// CHECK-32:    call void @llvm.ppc.isync()
// CHECK-32-NEXT:    ret void
//
void test_isync() {
  __isync();
}

// CHECK-LABEL: @test_lwsync(
// CHECK:    call void @llvm.ppc.lwsync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_lwsync(
// CHECK-32:    call void @llvm.ppc.lwsync()
// CHECK-32-NEXT:    ret void
//
void test_lwsync() {
  __lwsync();
}

// CHECK-LABEL: @test_iospace_lwsync(
// CHECK:    call void @llvm.ppc.iospace.lwsync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_iospace_lwsync(
// CHECK-32:    call void @llvm.ppc.iospace.lwsync()
// CHECK-32-NEXT:    ret void
//
void test_iospace_lwsync() {
  __iospace_lwsync();
}

// CHECK-LABEL: @test_sync(
// CHECK:    call void @llvm.ppc.sync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_sync(
// CHECK-32:    call void @llvm.ppc.sync()
// CHECK-32-NEXT:    ret void
//
void test_sync() {
  __sync();
}

// CHECK-LABEL: @test_iospace_sync(
// CHECK:    call void @llvm.ppc.iospace.sync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_iospace_sync(
// CHECK-32:    call void @llvm.ppc.iospace.sync()
// CHECK-32-NEXT:    ret void
//
void test_iospace_sync() {
  __iospace_sync();
}

// CHECK-LABEL: @test_dcbfl(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @b, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbfl(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_dcbfl(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @b, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbfl(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_dcbfl() {
  __dcbfl(b);
}

// CHECK-LABEL: @test_dcbflp(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @b, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbflp(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_dcbflp(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @b, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbflp(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_dcbflp() {
  __dcbflp(b);
}

// CHECK-LABEL: @test_dcbst(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @b, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbst(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_dcbst(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @b, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbst(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_dcbst() {
  __dcbst(b);
}

// CHECK-LABEL: @test_dcbt(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @c, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbt(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_dcbt(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @c, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbt(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_dcbt() {
  __dcbt(c);
}

// CHECK-LABEL: @test_dcbtst(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @c, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbtst(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_dcbtst(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @c, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbtst(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_dcbtst() {
  __dcbtst(c);
}

// CHECK-LABEL: @test_dcbz(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @c, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbz(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_dcbz(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @c, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbz(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_dcbz() {
  __dcbz(c);
}

// CHECK-LABEL: @test_builtin_ppc_popcntb(
// CHECK:    [[TMP0:%.*]] = load i64, ptr @a, align 8
// CHECK-NEXT:    [[POPCNTB:%.*]] = call i64 @llvm.ppc.popcntb.i64.i64(i64 [[TMP0]])
// CHECK-NEXT:    ret i64 [[POPCNTB]]
//
// CHECK-32-LABEL: @test_builtin_ppc_popcntb(
// CHECK-32:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-32-NEXT:    [[POPCNTB:%.*]] = call i32 @llvm.ppc.popcntb.i32.i32(i32 [[TMP0]])
// CHECK-32-NEXT:    ret i32 [[POPCNTB]]
//
unsigned long test_builtin_ppc_popcntb() {
  return __builtin_ppc_popcntb(a);
}

// CHECK-LABEL: @test_builtin_ppc_eieio(
// CHECK:    call void @llvm.ppc.eieio()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_eieio(
// CHECK-32:    call void @llvm.ppc.eieio()
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_eieio() {
  __builtin_ppc_eieio();
}

// CHECK-LABEL: @test_builtin_ppc_iospace_eieio(
// CHECK:    call void @llvm.ppc.iospace.eieio()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_iospace_eieio(
// CHECK-32:    call void @llvm.ppc.iospace.eieio()
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_iospace_eieio() {
  __builtin_ppc_iospace_eieio();
}

// CHECK-LABEL: @test_builtin_ppc_isync(
// CHECK:    call void @llvm.ppc.isync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_isync(
// CHECK-32:    call void @llvm.ppc.isync()
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_isync() {
  __builtin_ppc_isync();
}

// CHECK-LABEL: @test_builtin_ppc_lwsync(
// CHECK:    call void @llvm.ppc.lwsync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_lwsync(
// CHECK-32:    call void @llvm.ppc.lwsync()
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_lwsync() {
  __builtin_ppc_lwsync();
}

// CHECK-LABEL: @test_builtin_ppc_iospace_lwsync(
// CHECK:    call void @llvm.ppc.iospace.lwsync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_iospace_lwsync(
// CHECK-32:    call void @llvm.ppc.iospace.lwsync()
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_iospace_lwsync() {
  __builtin_ppc_iospace_lwsync();
}

// CHECK-LABEL: @test_builtin_ppc_sync(
// CHECK:    call void @llvm.ppc.sync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_sync(
// CHECK-32:    call void @llvm.ppc.sync()
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_sync() {
  __builtin_ppc_sync();
}

// CHECK-LABEL: @test_builtin_ppc_iospace_sync(
// CHECK:    call void @llvm.ppc.iospace.sync()
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_iospace_sync(
// CHECK-32:    call void @llvm.ppc.iospace.sync()
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_iospace_sync() {
  __builtin_ppc_iospace_sync();
}

// CHECK-LABEL: @test_builtin_ppc_dcbfl(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @b, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbfl(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_dcbfl(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @b, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbfl(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_dcbfl() {
  __builtin_ppc_dcbfl(b);
}

// CHECK-LABEL: @test_builtin_ppc_dcbflp(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @b, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbflp(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_dcbflp(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @b, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbflp(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_dcbflp() {
  __builtin_ppc_dcbflp(b);
}

// CHECK-LABEL: @test_builtin_ppc_dcbst(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @b, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbst(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_dcbst(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @b, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbst(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_dcbst() {
  __builtin_ppc_dcbst(b);
}

// CHECK-LABEL: @test_builtin_ppc_dcbt(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @c, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbt(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_dcbt(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @c, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbt(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_dcbt() {
  __builtin_ppc_dcbt(c);
}

// CHECK-LABEL: @test_builtin_ppc_dcbtst(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @c, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbtst(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_dcbtst(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @c, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbtst(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_dcbtst() {
  __builtin_ppc_dcbtst(c);
}

// CHECK-LABEL: @test_builtin_ppc_dcbz(
// CHECK:    [[TMP0:%.*]] = load ptr, ptr @c, align 8
// CHECK-NEXT:    call void @llvm.ppc.dcbz(ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
// CHECK-32-LABEL: @test_builtin_ppc_dcbz(
// CHECK-32:    [[TMP0:%.*]] = load ptr, ptr @c, align 4
// CHECK-32-NEXT:    call void @llvm.ppc.dcbz(ptr [[TMP0]])
// CHECK-32-NEXT:    ret void
//
void test_builtin_ppc_dcbz() {
  __builtin_ppc_dcbz(c);
}
