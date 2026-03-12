// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t-flat.cir
// RUN: FileCheck --input-file=%t-flat.cir %s --check-prefix=CIR-FLAT
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
  int data[5];
  void method();
  int &ref_return();
  void ref_param(int &i);
};

// Test that 'this' pointer attributes (nonnull, noundef, align,
// dereferenceable) are preserved on invoke instructions lowered from
// calls inside try blocks.
void test_this_ptr_attrs(S &s) {
  try {
    s.method();
  } catch (...) {}
}

// CIR-LABEL: cir.func {{.*}}@_Z19test_this_ptr_attrsR1S
// CIR:         cir.call @_ZN1S6methodEv({{%.*}}) : (!cir.ptr<!rec_S> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}) -> ()

// CIR-FLAT-LABEL: cir.func {{.*}}@_Z19test_this_ptr_attrsR1S
// CIR-FLAT:         cir.try_call @_ZN1S6methodEv({{%.*}}) ^{{bb[0-9]+}}, ^{{bb[0-9]+}}  : (!cir.ptr<!rec_S> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}) -> ()

// LLVM-LABEL: define {{.*}} void @_Z19test_this_ptr_attrsR1S
// LLVM:         invoke void @_ZN1S6methodEv(ptr noundef nonnull align 4 dereferenceable(20) {{%.*}})

// OGCG-LABEL: define {{.*}} void @_Z19test_this_ptr_attrsR1S
// OGCG:         invoke void @_ZN1S6methodEv(ptr noundef nonnull align 4 dereferenceable(20) {{%.*}})

// Test that both 'this' and reference parameter attributes are preserved
// on invoke instructions.
void test_ref_param_attrs(S &s, int &i) {
  try {
    s.ref_param(i);
  } catch (...) {}
}

// CIR-LABEL: cir.func {{.*}}@_Z20test_ref_param_attrsR1SRi
// CIR:         cir.call @_ZN1S9ref_paramERi({{%.*}}, {{%.*}}) : (!cir.ptr<!rec_S> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!s32i> {llvm.align = 8 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}) -> ()

// CIR-FLAT-LABEL: cir.func {{.*}}@_Z20test_ref_param_attrsR1SRi
// CIR-FLAT:         cir.try_call @_ZN1S9ref_paramERi({{%.*}}, {{%.*}}) ^{{bb[0-9]+}}, ^{{bb[0-9]+}} : (!cir.ptr<!rec_S> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!s32i> {llvm.align = 8 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}) -> ()

// LLVM-LABEL: define {{.*}} void @_Z20test_ref_param_attrsR1SRi
// LLVM:         invoke void @_ZN1S9ref_paramERi(ptr noundef nonnull align 4 dereferenceable(20) {{%.*}}, ptr noundef nonnull align 8 dereferenceable(4) {{%.*}})

// OGCG-LABEL: define {{.*}} void @_Z20test_ref_param_attrsR1SRi
// OGCG:         invoke void @_ZN1S9ref_paramERi(ptr noundef nonnull align 4 dereferenceable(20) {{%.*}}, ptr noundef nonnull align 4 dereferenceable(4) {{%.*}})

// Test that return attributes (nonnull, noundef, align, dereferenceable)
// and parameter attributes are preserved on invoke instructions that
// return a value.
void test_ref_return_attrs(S &s) {
  try {
    int &r = s.ref_return();
  } catch (...) {}
}

// CIR-LABEL: cir.func {{.*}}@_Z21test_ref_return_attrsR1S
// CIR:         cir.call @_ZN1S10ref_returnEv({{%.*}}) : (!cir.ptr<!rec_S> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}) -> (!cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef})

// CIR-FLAT-LABEL: cir.func {{.*}}@_Z21test_ref_return_attrsR1S
// CIR-FLAT:         cir.try_call @_ZN1S10ref_returnEv({{%.*}}) ^{{bb[0-9]+}}, ^{{bb[0-9]+}} : (!cir.ptr<!rec_S> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}) -> (!cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef})

// LLVM-LABEL: define {{.*}} void @_Z21test_ref_return_attrsR1S
// LLVM:         {{%.*}} = invoke noundef nonnull align 4 dereferenceable(4) ptr @_ZN1S10ref_returnEv(ptr noundef nonnull align 4 dereferenceable(20) {{%.*}})

// OGCG-LABEL: define {{.*}} void @_Z21test_ref_return_attrsR1S
// OGCG:         {{%.*}} = invoke noundef nonnull align 4 dereferenceable(4) ptr @_ZN1S10ref_returnEv(ptr noundef nonnull align 4 dereferenceable(20) {{%.*}})
