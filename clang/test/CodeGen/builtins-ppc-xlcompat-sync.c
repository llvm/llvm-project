// RUN: %clang_cc1 -triple powerpc64-unknown-unknown \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s

void test_popcntb() {
// CHECK-LABEL: @test_popcntb(
// CHECK-NEXT:  entry:
 
  unsigned long a;
  unsigned long b = __popcntb(a);
// CHECK: %1 = call i64 @llvm.ppc.popcntb(i64 %0)
}

void test_eieio() {
// CHECK-LABEL: @test_eieio(
// CHECK-NEXT: entry:

  __eieio();
// CHECK: call void @llvm.ppc.eieio()
}

void test_iospace_eieio() {
// CHECK-LABEL: @test_iospace_eieio(
// CHECK-NEXT: entry:

  __iospace_eieio();
// CHECK: call void @llvm.ppc.iospace.eieio()
}

void test_isync() {
// CHECK-LABEL: @test_isync(
// CHECK-NEXT: entry:

  __isync();
// CHECK: call void @llvm.ppc.isync()
}

void test_lwsync() {
// CHECK-LABEL: @test_lwsync(
// CHECK-NEXT: entry:

  __lwsync();
// CHECK: call void @llvm.ppc.lwsync()
}

void test_iospace_lwsync() {
// CHECK-LABEL: @test_iospace_lwsync(
// CHECK-NEXT: entry:

  __iospace_lwsync();
// CHECK: call void @llvm.ppc.iospace.lwsync()
}

void test_sync() {
// CHECK-LABEL: @test_sync(
// CHECK-NEXT: entry:

  __sync();
// CHECK: call void @llvm.ppc.sync()
}

void test_iospace_sync() {
// CHECK-LABEL: @test_iospace_sync(
// CHECK-NEXT: entry:

  __iospace_sync();
// CHECK: call void @llvm.ppc.iospace.sync()  
}

void test_dcbfl() {
// CHECK-LABEL: @test_dcbfl(
// CHECK-NEXT: entry:

  const void* a;
  __dcbfl(a);
// CHECK: call void @llvm.ppc.dcbfl(i8* %0)
}

void test_dcbflp() {
// CHECK-LABEL: @test_dcbflp(
// CHECK-NEXT: entry:

  const void* a;
  __dcbflp(a);
// CHECK: call void @llvm.ppc.dcbflp(i8* %0)
}

void test_dcbst() {
// CHECK-LABEL: @test_dcbst(
// CHECK-NEXT: entry:

  const void* a;
  __dcbst(a);
// CHECK: call void @llvm.ppc.dcbst(i8* %0)
}

void test_dcbt() {
// CHECK-LABEL: @test_dcbt(
// CHECK-NEXT: entry:

  void* a;
  __dcbt(a);
// CHECK: call void @llvm.ppc.dcbt(i8* %0)
}

void test_dcbtst() {
// CHECK-LABEL: @test_dcbtst(
// CHECK-NEXT: entry:

  void* a;
  __dcbtst(a);
// CHECK: call void @llvm.ppc.dcbtst(i8* %0)
}

void test_dcbz() {
// CHECK-LABEL: @test_dcbz(
// CHECK-NEXT: entry:

  void* a;
  __dcbz(a);
// CHECK: call void @llvm.ppc.dcbz(i8* %0)
}

void test_icbt() {
// CHECK-LABEL: @test_icbt(
// CHECK-NEXT: entry:

  void* a;
  __icbt(a);
// CHECK: call void @llvm.ppc.icbt(i8* %0)
}

void test_builtin_ppc_popcntb() {
// CHECK-LABEL: @test_builtin_ppc_popcntb(
// CHECK-NEXT:  entry:
 
  unsigned long a;
  unsigned long b = __builtin_ppc_popcntb(a);
// CHECK: %1 = call i64 @llvm.ppc.popcntb(i64 %0)
}

void test_builtin_ppc_eieio() {
// CHECK-LABEL: @test_builtin_ppc_eieio(
// CHECK-NEXT: entry:

  __builtin_ppc_eieio();
// CHECK: call void @llvm.ppc.eieio()
}

void test_builtin_ppc_iospace_eieio() {
// CHECK-LABEL: @test_builtin_ppc_iospace_eieio(
// CHECK-NEXT: entry:

  __builtin_ppc_iospace_eieio();
// CHECK: call void @llvm.ppc.iospace.eieio()
}

void test_builtin_ppc_isync() {
// CHECK-LABEL: @test_builtin_ppc_isync(
// CHECK-NEXT: entry:

  __builtin_ppc_isync();
// CHECK: call void @llvm.ppc.isync()
}

void test_builtin_ppc_lwsync() {
// CHECK-LABEL: @test_builtin_ppc_lwsync(
// CHECK-NEXT: entry:

  __builtin_ppc_lwsync();
// CHECK: call void @llvm.ppc.lwsync()
}

void test_builtin_ppc_iospace_lwsync() {
// CHECK-LABEL: @test_builtin_ppc_iospace_lwsync(
// CHECK-NEXT: entry:

  __builtin_ppc_iospace_lwsync();
// CHECK: call void @llvm.ppc.iospace.lwsync()
}

void test_builtin_ppc_sync() {
// CHECK-LABEL: @test_builtin_ppc_sync(
// CHECK-NEXT: entry:

  __builtin_ppc_sync();
// CHECK: call void @llvm.ppc.sync()
}

void test_builtin_ppc_iospace_sync() {
// CHECK-LABEL: @test_builtin_ppc_iospace_sync(
// CHECK-NEXT: entry:

  __builtin_ppc_iospace_sync();
// CHECK: call void @llvm.ppc.iospace.sync()  
}

void test_builtin_ppc_dcbfl() {
// CHECK-LABEL: @test_builtin_ppc_dcbfl(
// CHECK-NEXT: entry:

  const void* a;
  __builtin_ppc_dcbfl(a);
// CHECK: call void @llvm.ppc.dcbfl(i8* %0)
}

void test_builtin_ppc_dcbflp() {
// CHECK-LABEL: @test_builtin_ppc_dcbflp(
// CHECK-NEXT: entry:

  const void* a;
  __builtin_ppc_dcbflp(a);
// CHECK: call void @llvm.ppc.dcbflp(i8* %0)
}

void test_builtin_ppc_dcbst() {
// CHECK-LABEL: @test_builtin_ppc_dcbst(
// CHECK-NEXT: entry:

  const void* a;
  __builtin_ppc_dcbst(a);
// CHECK: call void @llvm.ppc.dcbst(i8* %0)
}

void test_builtin_ppc_dcbt() {
// CHECK-LABEL: @test_builtin_ppc_dcbt(
// CHECK-NEXT: entry:

  void* a;
  __builtin_ppc_dcbt(a);
// CHECK: call void @llvm.ppc.dcbt(i8* %0)
}

void test_builtin_ppc_dcbtst() {
// CHECK-LABEL: @test_builtin_ppc_dcbtst(
// CHECK-NEXT: entry:

  void* a;
  __builtin_ppc_dcbtst(a);
// CHECK: call void @llvm.ppc.dcbtst(i8* %0)
}

void test_builtin_ppc_dcbz() {
// CHECK-LABEL: @test_builtin_ppc_dcbz(
// CHECK-NEXT: entry:

  void* a;
  __builtin_ppc_dcbz(a);
// CHECK: call void @llvm.ppc.dcbz(i8* %0)
}

void test_builtin_ppc_icbt() {
// CHECK-LABEL: @test_builtin_ppc_icbt(
// CHECK-NEXT: entry:

  void* a;
  __builtin_ppc_icbt(a);
// CHECK: call void @llvm.ppc.icbt(i8* %0)
}
