// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 %s -x hip -fcuda-is-device -emit-llvm -O0 -o - \
// RUN:   -triple=amdgcn-amd-amdhsa  | opt -S | FileCheck %s

__attribute__((device)) void test_non_volatile_parameter32(int *ptr) {
  // CHECK-LABEL: test_non_volatile_parameter32
  int res;
  // CHECK: %ptr.addr = alloca i32*, align 8, addrspace(5)
  // CHECK-NEXT: %ptr.addr.ascast = addrspacecast i32* addrspace(5)* %ptr.addr to i32**
  // CHECK-NEXT: %res = alloca i32, align 4, addrspace(5)
  // CHECK-NEXT: %res.ascast = addrspacecast i32 addrspace(5)* %res to i32*
  // CHECK-NEXT: store i32* %ptr, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %0 = load i32*, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %1 = load i32*, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %2 = load i32, i32* %1, align 4
  // CHECK-NEXT: %3 = call i32 @llvm.amdgcn.atomic.inc.i32.p0i32(i32* %0, i32 %2, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i32 %3, i32* %res.ascast, align 4
  res = __builtin_amdgcn_atomic_inc32(ptr, *ptr, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %4 = load i32*, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %5 = load i32*, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %6 = load i32, i32* %5, align 4
  // CHECK-NEXT: %7 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* %4, i32 %6, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i32 %7, i32* %res.ascast, align 4
  res = __builtin_amdgcn_atomic_dec32(ptr, *ptr, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((device)) void test_non_volatile_parameter64(__INT64_TYPE__ *ptr) {
  // CHECK-LABEL: test_non_volatile_parameter64
  __INT64_TYPE__ res;
  // CHECK: %ptr.addr = alloca i64*, align 8, addrspace(5)
  // CHECK-NEXT: %ptr.addr.ascast = addrspacecast i64* addrspace(5)* %ptr.addr to i64**
  // CHECK-NEXT: %res = alloca i64, align 8, addrspace(5)
  // CHECK-NEXT: %res.ascast = addrspacecast i64 addrspace(5)* %res to i64*
  // CHECK-NEXT: store i64* %ptr, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %0 = load i64*, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %1 = load i64*, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %2 = load i64, i64* %1, align 8
  // CHECK-NEXT: %3 = call i64 @llvm.amdgcn.atomic.inc.i64.p0i64(i64* %0, i64 %2, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i64 %3, i64* %res.ascast, align 8
  res = __builtin_amdgcn_atomic_inc64(ptr, *ptr, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %4 = load i64*, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %5 = load i64*, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %6 = load i64, i64* %5, align 8
  // CHECK-NEXT: %7 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* %4, i64 %6, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i64 %7, i64* %res.ascast, align 8
  res = __builtin_amdgcn_atomic_dec64(ptr, *ptr, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((device)) void test_volatile_parameter32(volatile int *ptr) {
  // CHECK-LABEL: test_volatile_parameter32
  int res;
  // CHECK: %ptr.addr = alloca i32*, align 8, addrspace(5)
  // CHECK-NEXT: %ptr.addr.ascast = addrspacecast i32* addrspace(5)* %ptr.addr to i32**
  // CHECK-NEXT: %res = alloca i32, align 4, addrspace(5)
  // CHECK-NEXT: %res.ascast = addrspacecast i32 addrspace(5)* %res to i32*
  // CHECK-NEXT: store i32* %ptr, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %0 = load i32*, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %1 = load i32*, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %2 = load volatile i32, i32* %1, align 4
  // CHECK-NEXT: %3 = call i32 @llvm.amdgcn.atomic.inc.i32.p0i32(i32* %0, i32 %2, i32 7, i32 2, i1 true)
  // CHECK-NEXT: store i32 %3, i32* %res.ascast, align 4
  res = __builtin_amdgcn_atomic_inc32(ptr, *ptr, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %4 = load i32*, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %5 = load i32*, i32** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %6 = load volatile i32, i32* %5, align 4
  // CHECK-NEXT: %7 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* %4, i32 %6, i32 7, i32 2, i1 true)
  // CHECK-NEXT: store i32 %7, i32* %res.ascast, align 4
  res = __builtin_amdgcn_atomic_dec32(ptr, *ptr, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((device)) void test_volatile_parameter64(volatile __INT64_TYPE__ *ptr) {
  // CHECK-LABEL: test_volatile_parameter64
  __INT64_TYPE__ res;
  // CHECK: %ptr.addr = alloca i64*, align 8, addrspace(5)
  // CHECK-NEXT: %ptr.addr.ascast = addrspacecast i64* addrspace(5)* %ptr.addr to i64**
  // CHECK-NEXT: %res = alloca i64, align 8, addrspace(5)
  // CHECK-NEXT: %res.ascast = addrspacecast i64 addrspace(5)* %res to i64*
  // CHECK-NEXT: store i64* %ptr, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %0 = load i64*, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %1 = load i64*, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %2 = load volatile i64, i64* %1, align 8
  // CHECK-NEXT: %3 = call i64 @llvm.amdgcn.atomic.inc.i64.p0i64(i64* %0, i64 %2, i32 7, i32 2, i1 true)
  // CHECK-NEXT: store i64 %3, i64* %res.ascast, align 8
  res = __builtin_amdgcn_atomic_inc64(ptr, *ptr, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %4 = load i64*, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %5 = load i64*, i64** %ptr.addr.ascast, align 8
  // CHECK-NEXT: %6 = load volatile i64, i64* %5, align 8
  // CHECK-NEXT: %7 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* %4, i64 %6, i32 7, i32 2, i1 true)
  // CHECK-NEXT: store i64 %7, i64* %res.ascast, align 8
  res = __builtin_amdgcn_atomic_dec64(ptr, *ptr, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((device)) void test_shared32() {
  // CHECK-LABEL: test_shared32
  __attribute__((shared)) int val;

  // CHECK: %0 = load i32, i32* addrspacecast (i32 addrspace(3)* @_ZZ13test_shared32vE3val to i32*), align 4
  // CHECK-NEXT: %1 = call i32 @llvm.amdgcn.atomic.inc.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ13test_shared32vE3val to i32*), i32 %0, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i32 %1, i32* addrspacecast (i32 addrspace(3)* @_ZZ13test_shared32vE3val to i32*), align 4
  val = __builtin_amdgcn_atomic_inc32(&val, val, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %2 = load i32, i32* addrspacecast (i32 addrspace(3)* @_ZZ13test_shared32vE3val to i32*), align 4
  // CHECK-NEXT: %3 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ13test_shared32vE3val to i32*), i32 %2, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i32 %3, i32* addrspacecast (i32 addrspace(3)* @_ZZ13test_shared32vE3val to i32*), align 4
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((device)) void test_shared64() {
  // CHECK-LABEL: test_shared64
  __attribute__((shared)) __INT64_TYPE__ val;

  // CHECK: %0 = load i64, i64* addrspacecast (i64 addrspace(3)* @_ZZ13test_shared64vE3val to i64*), align 8
  // CHECK-NEXT: %1 = call i64 @llvm.amdgcn.atomic.inc.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ13test_shared64vE3val to i64*), i64 %0, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i64 %1, i64* addrspacecast (i64 addrspace(3)* @_ZZ13test_shared64vE3val to i64*), align 8
  val = __builtin_amdgcn_atomic_inc64(&val, val, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %2 = load i64, i64* addrspacecast (i64 addrspace(3)* @_ZZ13test_shared64vE3val to i64*), align 8
  // CHECK-NEXT: %3 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ13test_shared64vE3val to i64*), i64 %2, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i64 %3, i64* addrspacecast (i64 addrspace(3)* @_ZZ13test_shared64vE3val to i64*), align 8
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_SEQ_CST, "workgroup");
}

int global_val32;
__attribute__((device)) void test_global32() {
  // CHECK-LABEL: test_global32
  // CHECK: %0 = load i32, i32* addrspacecast (i32 addrspace(1)* @global_val32 to i32*), align 4
  // CHECK-NEXT: %1 = call i32 @llvm.amdgcn.atomic.inc.i32.p0i32(i32* addrspacecast (i32 addrspace(1)* @global_val32 to i32*), i32 %0, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i32 %1, i32* addrspacecast (i32 addrspace(1)* @global_val32 to i32*), align 4
  global_val32 = __builtin_amdgcn_atomic_inc32(&global_val32, global_val32, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %2 = load i32, i32* addrspacecast (i32 addrspace(1)* @global_val32 to i32*), align 4
  // CHECK-NEXT: %3 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* addrspacecast (i32 addrspace(1)* @global_val32 to i32*), i32 %2, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i32 %3, i32* addrspacecast (i32 addrspace(1)* @global_val32 to i32*), align 4
  global_val32 = __builtin_amdgcn_atomic_dec32(&global_val32, global_val32, __ATOMIC_SEQ_CST, "workgroup");
}

__INT64_TYPE__ global_val64;
__attribute__((device)) void test_global64() {
  // CHECK-LABEL: test_global64
  // CHECK: %0 = load i64, i64* addrspacecast (i64 addrspace(1)* @global_val64 to i64*), align 8
  // CHECK-NEXT: %1 = call i64 @llvm.amdgcn.atomic.inc.i64.p0i64(i64* addrspacecast (i64 addrspace(1)* @global_val64 to i64*), i64 %0, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i64 %1, i64* addrspacecast (i64 addrspace(1)* @global_val64 to i64*), align 8
  global_val64 = __builtin_amdgcn_atomic_inc64(&global_val64, global_val64, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %2 = load i64, i64* addrspacecast (i64 addrspace(1)* @global_val64 to i64*), align 8
  // CHECK-NEXT: %3 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* addrspacecast (i64 addrspace(1)* @global_val64 to i64*), i64 %2, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i64 %3, i64* addrspacecast (i64 addrspace(1)* @global_val64 to i64*), align 8
  global_val64 = __builtin_amdgcn_atomic_dec64(&global_val64, global_val64, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((constant)) int cval32;
__attribute__((device)) void test_constant32() {
  // CHECK-LABEL: test_constant32
  int local_val;

  // CHECK: %0 = load i32, i32* addrspacecast (i32 addrspace(4)* @cval32 to i32*), align 4
  // CHECK-NEXT: %1 = call i32 @llvm.amdgcn.atomic.inc.i32.p0i32(i32* addrspacecast (i32 addrspace(4)* @cval32 to i32*), i32 %0, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i32 %1, i32* %local_val.ascast, align 4
  local_val = __builtin_amdgcn_atomic_inc32(&cval32, cval32, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %2 = load i32, i32* addrspacecast (i32 addrspace(4)* @cval32 to i32*), align 4
  // CHECK-NEXT: %3 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* addrspacecast (i32 addrspace(4)* @cval32 to i32*), i32 %2, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i32 %3, i32* %local_val.ascast, align 4
  local_val = __builtin_amdgcn_atomic_dec32(&cval32, cval32, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((constant)) __INT64_TYPE__ cval64;
__attribute__((device)) void test_constant64() {
  // CHECK-LABEL: test_constant64
  __INT64_TYPE__ local_val;

  // CHECK: %0 = load i64, i64* addrspacecast (i64 addrspace(4)* @cval64 to i64*), align 8
  // CHECK-NEXT: %1 = call i64 @llvm.amdgcn.atomic.inc.i64.p0i64(i64* addrspacecast (i64 addrspace(4)* @cval64 to i64*), i64 %0, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i64 %1, i64* %local_val.ascast, align 8
  local_val = __builtin_amdgcn_atomic_inc64(&cval64, cval64, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %2 = load i64, i64* addrspacecast (i64 addrspace(4)* @cval64 to i64*), align 8
  // CHECK-NEXT: %3 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* addrspacecast (i64 addrspace(4)* @cval64 to i64*), i64 %2, i32 7, i32 2, i1 false)
  // CHECK-NEXT: store i64 %3, i64* %local_val.ascast, align 8
  local_val = __builtin_amdgcn_atomic_dec64(&cval64, cval64, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((device)) void test_order32() {
  // CHECK-LABEL: test_order32
  __attribute__((shared)) int val;

  // CHECK: %1 = call i32 @llvm.amdgcn.atomic.inc.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ12test_order32vE3val to i32*), i32 %0, i32 4, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_inc32(&val, val, __ATOMIC_ACQUIRE, "workgroup");

  // CHECK: %3 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ12test_order32vE3val to i32*), i32 %2, i32 5, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_RELEASE, "workgroup");

  // CHECK: %5 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ12test_order32vE3val to i32*), i32 %4, i32 6, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_ACQ_REL, "workgroup");

  // CHECK: %7 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ12test_order32vE3val to i32*), i32 %6, i32 7, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((device)) void test_order64() {
  // CHECK-LABEL: test_order64
  __attribute__((shared)) __INT64_TYPE__ val;

  // CHECK: %1 = call i64 @llvm.amdgcn.atomic.inc.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ12test_order64vE3val to i64*), i64 %0, i32 4, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_inc64(&val, val, __ATOMIC_ACQUIRE, "workgroup");

  // CHECK: %3 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ12test_order64vE3val to i64*), i64 %2, i32 5, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_RELEASE, "workgroup");

  // CHECK: %5 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ12test_order64vE3val to i64*), i64 %4, i32 6, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_ACQ_REL, "workgroup");

  // CHECK: %7 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ12test_order64vE3val to i64*), i64 %6, i32 7, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_SEQ_CST, "workgroup");
}

__attribute__((device)) void test_scope32() {
  // CHECK-LABEL: test_scope32
  __attribute__((shared)) int val;

  // CHECK: %1 = call i32 @llvm.amdgcn.atomic.inc.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ12test_scope32vE3val to i32*), i32 %0, i32 7, i32 1, i1 false)
  val = __builtin_amdgcn_atomic_inc32(&val, val, __ATOMIC_SEQ_CST, "");

  // CHECK: %3 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ12test_scope32vE3val to i32*), i32 %2, i32 7, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %5 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ12test_scope32vE3val to i32*), i32 %4, i32 7, i32 3, i1 false)
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_SEQ_CST, "agent");

  // CHECK: %7 = call i32 @llvm.amdgcn.atomic.dec.i32.p0i32(i32* addrspacecast (i32 addrspace(3)* @_ZZ12test_scope32vE3val to i32*), i32 %6, i32 7, i32 4, i1 false)
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_SEQ_CST, "wavefront");
}

__attribute__((device)) void test_scope64() {
  // CHECK-LABEL: test_scope64
  __attribute__((shared)) __INT64_TYPE__ val;

  // CHECK: %1 = call i64 @llvm.amdgcn.atomic.inc.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ12test_scope64vE3val to i64*), i64 %0, i32 7, i32 1, i1 false)
  val = __builtin_amdgcn_atomic_inc64(&val, val, __ATOMIC_SEQ_CST, "");

  // CHECK: %3 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ12test_scope64vE3val to i64*), i64 %2, i32 7, i32 2, i1 false)
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_SEQ_CST, "workgroup");

  // CHECK: %5 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ12test_scope64vE3val to i64*), i64 %4, i32 7, i32 3, i1 false)
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_SEQ_CST, "agent");

  // CHECK: %7 = call i64 @llvm.amdgcn.atomic.dec.i64.p0i64(i64* addrspacecast (i64 addrspace(3)* @_ZZ12test_scope64vE3val to i64*), i64 %6, i32 7, i32 4, i1 false)
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_SEQ_CST, "wavefront");
}
