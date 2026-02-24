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
  // CIR: cir.func {{.*}}@_ZN6Struct9this_funcEv(%{{.*}}: !cir.ptr<!rec_Struct> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef} {{.*}}) {
  // BOTH: define {{.*}}void @_ZN6Struct9this_funcEv(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}})
void Struct::arg_attr(Struct s, int &i, Incomplete &j){}
  // CIR: cir.func {{.*}}@_ZN6Struct8arg_attrES_RiR10Incomplete(%{{.*}}: !cir.ptr<!rec_Struct> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef} {{.*}}, %{{.*}}: !rec_Struct {{.*}}, %{{.*}}: !cir.ptr<!s32i> {llvm.align = 8 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef} {{.*}}, %arg3: !cir.ptr<!rec_Incomplete> {llvm.align = 8 : i64, llvm.nonnull, llvm.noundef} {{.*}}) {
  // LLVM: define {{.*}}void @_ZN6Struct8arg_attrES_RiR10Incomplete(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}}, %struct.Struct %{{.*}}, ptr noundef nonnull align 8 dereferenceable(4) %{{.*}}, ptr noundef nonnull align 8 %{{.*}})
  // OGCG: define {{.*}}void @_ZN6Struct8arg_attrES_RiR10Incomplete(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}}, ptr noundef byval(%struct.Struct) align 8 %{{.*}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.*}}, ptr noundef nonnull align 1 %{{.*}})

void caller(Struct s, int i, Incomplete &inc) {

  s.this_func();
  // CIR: cir.call @_ZN6Struct9this_funcEv(%{{.*}}) : (!cir.ptr<!rec_Struct> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef})
  // BOTH: call void @_ZN6Struct9this_funcEv(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}})
  s.arg_attr(s, i, inc);
  // CIR: cir.call @_ZN6Struct8arg_attrES_RiR10Incomplete(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!cir.ptr<!rec_Struct> {llvm.align = 4 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}, !rec_Struct, !cir.ptr<!s32i> {llvm.align = 8 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!rec_Incomplete> {llvm.align = 8 : i64, llvm.nonnull, llvm.noundef})
  // LLVM: call void @_ZN6Struct8arg_attrES_RiR10Incomplete(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}}, %struct.Struct %{{.*}}, ptr noundef nonnull align 8 dereferenceable(4) %{{.*}}, ptr noundef nonnull align 8 %{{.*}})
  // OGCG: call void @_ZN6Struct8arg_attrES_RiR10Incomplete(ptr noundef nonnull align 4 dereferenceable(20) %{{.*}}, ptr noundef byval(%struct.Struct) align 8 %{{.*}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.*}}, ptr noundef nonnull align 1 %{{.*}})
}

