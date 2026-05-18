// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=BOTH,LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=BOTH,OGCG
struct Incomplete;

struct Struct {
  void this_func();

  void arg_attr(Struct s, int &i, Incomplete &j);
  // If this doesn't have data, it get optimized away in classic codegen.
  int hasData[5];
};

void Struct::this_func(){}
  // CIR: cir.func {{.*}}@_ZN6Struct9this_funcEv(%{{.*}}: !cir.ptr<!rec_Struct> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef} {{.*}}) {{.*}} {
  // BOTH: define {{.*}}void @_ZN6Struct9this_funcEv(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}})
void Struct::arg_attr(Struct s, int &i, Incomplete &j){}
  // CIR: cir.func {{.*}}@_ZN6Struct8arg_attrES_RiR10Incomplete(%{{.*}}: !cir.ptr<!rec_Struct> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef} {{.*}}, %{{.*}}: !rec_Struct {{.*}}, %{{.*}}: !cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef} {{.*}}, %arg3: !cir.ptr<!rec_Incomplete> {llvm.align = 1 : i64, llvm.nonnull, llvm.noundef} {{.*}}) {{.*}} {
  // LLVM: define {{.*}}void @_ZN6Struct8arg_attrES_RiR10Incomplete(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}}, %struct.Struct %{{.*}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.*}}, ptr noundef nonnull align 1 %{{.*}})
  // OGCG: define {{.*}}void @_ZN6Struct8arg_attrES_RiR10Incomplete(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}}, ptr noundef byval(%struct.Struct) align 8 %{{.*}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.*}}, ptr noundef nonnull align 1 %{{.*}})

struct __attribute__((aligned(32))) Aligned32 {
  int x;
  void method();
};

void Aligned32::method() {}
  // CIR: cir.func {{.*}}@_ZN9Aligned326methodEv(%{{.*}}: !cir.ptr<!rec_Aligned32> {llvm.align = 32 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef} {{.*}}) {{.*}} {
  // BOTH: define {{.*}}void @_ZN9Aligned326methodEv(ptr noundef nonnull align 32 dereferenceable(32) %{{.*}})

void aligned_ref(Aligned32 &a) {}
  // CIR: cir.func {{.*}}@_Z11aligned_refR9Aligned32(%{{.*}}: !cir.ptr<!rec_Aligned32> {llvm.align = 32 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef} {{.*}}) {{.*}} {
  // BOTH: define {{.*}}void @_Z11aligned_refR9Aligned32(ptr noundef nonnull align 32 dereferenceable(32) %{{.*}})

int g;
int &return_int_ref() { return g; }
  // CIR: cir.func {{.*}}@_Z14return_int_refv() -> (!cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}) {{.*}} {
  // BOTH: define {{.*}}noundef nonnull align 4 dereferenceable(4) ptr @_Z14return_int_refv()

Aligned32 ga;
Aligned32 &return_aligned_ref() { return ga; }
  // CIR: cir.func {{.*}}@_Z18return_aligned_refv() -> (!cir.ptr<!rec_Aligned32> {llvm.align = 32 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) {{.*}} {
  // BOTH: define {{.*}}noundef nonnull align 32 dereferenceable(32) ptr @_Z18return_aligned_refv()

void caller(Struct s, int i, Incomplete &inc) {

  s.this_func();
  // CIR: cir.call @_ZN6Struct9this_funcEv(%{{.*}}) : (!cir.ptr<!rec_Struct> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef})
  // BOTH: call void @_ZN6Struct9this_funcEv(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}})
  s.arg_attr(s, i, inc);
  // CIR: cir.call @_ZN6Struct8arg_attrES_RiR10Incomplete(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!cir.ptr<!rec_Struct> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}, !rec_Struct, !cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!rec_Incomplete> {llvm.align = 1 : i64, llvm.nonnull, llvm.noundef})
  // LLVM: call void @_ZN6Struct8arg_attrES_RiR10Incomplete(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}}, %struct.Struct %{{.*}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.*}}, ptr noundef nonnull align 1 %{{.*}})
  // OGCG: call void @_ZN6Struct8arg_attrES_RiR10Incomplete(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}}, ptr noundef byval(%struct.Struct) align 8 %{{.*}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.*}}, ptr noundef nonnull align 1 %{{.*}})
}

