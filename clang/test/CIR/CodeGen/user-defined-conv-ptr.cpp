// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Resource {
  int value;
};

template <typename T>
struct Ptr {
  T *p;
  operator T*() const { return p; }
  T** operator&() { return &p; }
};

void use_ptr(Resource *r);
void use_void_pp(void **pp);

// Test user-defined conversion operator producing a pointer
// (CK_UserDefinedConversion through emitPointerWithAlignment).
void test_conv_op(Ptr<Resource> smart) {
  use_ptr(smart);
}

// CIR-LABEL: @_Z12test_conv_op3PtrI8ResourceE
// CIR:         cir.call @_ZNK3PtrI8ResourceEcvPS0_Ev
// CIR-NEXT:    cir.call @_Z7use_ptrP8Resource

// LLVM-LABEL: define dso_local void @_Z12test_conv_op3PtrI8ResourceE
// LLVM:         call noundef ptr @_ZNK3PtrI8ResourceEcvPS0_Ev
// LLVM-NEXT:    call void @_Z7use_ptrP8Resource

// OGCG-LABEL: define dso_local void @_Z12test_conv_op3PtrI8ResourceE
// OGCG:         call noundef ptr @_ZNK3PtrI8ResourceEcvPS0_Ev
// OGCG-NEXT:    call void @_Z7use_ptrP8Resource

// Test operator& returning pointer-to-pointer, cast to void**.
void test_addr_op(Ptr<Resource> smart) {
  use_void_pp((void **)&smart);
}

// CIR-LABEL: @_Z12test_addr_op3PtrI8ResourceE
// CIR:         cir.call @_ZN3PtrI8ResourceEadEv
// CIR:         cir.cast bitcast
// CIR:         cir.call @_Z11use_void_ppPPv

// LLVM-LABEL: define dso_local void @_Z12test_addr_op3PtrI8ResourceE
// LLVM:         call noundef ptr @_ZN3PtrI8ResourceEadEv
// LLVM:         call void @_Z11use_void_ppPPv

// OGCG-LABEL: define dso_local void @_Z12test_addr_op3PtrI8ResourceE
// OGCG:         call noundef ptr @_ZN3PtrI8ResourceEadEv
// OGCG:         call void @_Z11use_void_ppPPv
