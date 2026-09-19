// Test for issue - (https://github.com/llvm/llvm-project/issues/222931)
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++11 -o - -fcxx-exceptions -fexceptions -fms-extensions | FileCheck -check-prefix=CHECK-PTR32 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++11 -o - -fcxx-exceptions -fexceptions -fms-extensions | FileCheck -check-prefix=CHECK-AS270-BYREF %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++11 -o - -fcxx-exceptions -fexceptions -fms-extensions | FileCheck -check-prefix=CHECK-AS270-REF %s

void sink_ptr32(int);

void test_ptr32() {
  try { throw (int * __ptr32)0; } catch (int * __ptr32 p) { sink_ptr32(*p); }
}

// CHECK-PTR32: %exn.casted = addrspacecast ptr %{{.*}} to ptr addrspace(270)

typedef int __attribute__((address_space(270))) as270_int;
void sink_as270(as270_int &);

void test_byref() {
  try { throw (as270_int)0; } catch (as270_int &p) { sink_as270(p); }
}

// CHECK-AS270-BYREF: %exn.byref = addrspacecast ptr {{.*}} to ptr addrspace(270)

struct S2 {};
typedef S2 __attribute__((address_space(270))) *PtrRec;
void sink_ptrrec(PtrRec &);

void test_ptrrec() {
  try { throw (PtrRec)0; } catch (PtrRec &p) { sink_ptrrec(p); }
}

// CHECK-AS270-REF: store ptr addrspace(270) %{{.*}}, ptr %exn.byref.tmp, align 8