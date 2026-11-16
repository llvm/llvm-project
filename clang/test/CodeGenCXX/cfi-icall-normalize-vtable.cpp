// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=kcfi -fwhole-program-vtables -flto=thin -flto-unit -fsanitize-cfi-icall-experimental-normalize-integers -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,KCFI --implicit-check-not=.normalized.virtual
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall,cfi-vcall,cfi-mfcall -fsanitize-trap=cfi-icall,cfi-vcall,cfi-mfcall -fwhole-program-vtables -flto=thin -flto-unit -fsanitize-cfi-icall-experimental-normalize-integers -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CFI,CFI-NOCG,CFI-PUBLIC --implicit-check-not=.normalized.virtual
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-icall,cfi-vcall,cfi-mfcall -fsanitize-trap=cfi-icall,cfi-vcall,cfi-mfcall -fwhole-program-vtables -flto=thin -flto-unit -fsanitize-cfi-icall-experimental-normalize-integers -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CFI,CFI-NOCG,CFI-HIDDEN --implicit-check-not=.normalized.virtual
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section -fsanitize=cfi-icall,cfi-vcall,cfi-mfcall -fsanitize-trap=cfi-icall,cfi-vcall,cfi-mfcall -fwhole-program-vtables -flto=thin -flto-unit -fsanitize-cfi-icall-experimental-normalize-integers -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CFI,CFI-PUBLIC,CALLGRAPH --implicit-check-not=.normalized.virtual

// Verify that -fsanitize-cfi-icall-experimental-normalize-integers only
// normalizes cfi-icall/kcfi function metadata and does not alter class vtable
// !type metadata, member-pointer metadata, or -fexperimental-call-graph-section
// metadata.

struct Base {
  virtual int get(int x);
};

struct Derived : Base {
  int get(int x) override;
};

template <typename T>
struct Box {
  virtual T val(T x);
};

int Base::get(int x) { return x; }
int Derived::get(int x) { return x + 1; }
template <typename T>
T Box<T>::val(T x) { return x; }
template struct Box<int>;

void icall_target(int x) {}

int test_calls(Base *b, int (Base::*mp)(int), Box<int> *box, void (*fp)(int), int x) {
  int r = b->get(x);
  r += (b->*mp)(x);
  r += box->val(x);
  fp(x);
  return r;
}

void take_addr() {
  void (*fp)(int) = &icall_target;
  fp(1);
}

// CHECK-NOT: .normalized.virtual
// CHECK-NOT: _ZTS4Base.normalized
// CHECK-NOT: _ZTS7Derived.normalized
// CHECK-NOT: _ZTS3BoxIu3i32E
// CHECK-NOT: _ZTS3BoxIiE.normalized
// CHECK-NOT: _ZTSM4BaseFuu3i32E
// CHECK-NOT: _ZTSM4BaseFiiE.normalized.virtual

// CHECK-DAG: @_ZTV4Base = {{.*}}!type ![[BASE_TYPE:[0-9]+]], !type ![[BASE_VMEMPTR:[0-9]+]]
// CHECK-DAG: @_ZTV7Derived = {{.*}}!type ![[BASE_TYPE]], !type ![[BASE_VMEMPTR]], !type ![[DERIVED_TYPE:[0-9]+]], !type ![[DERIVED_VMEMPTR:[0-9]+]]
// CHECK-DAG: @_ZTV3BoxIiE = {{.*}}!type ![[BOX_TYPE:[0-9]+]], !type ![[BOX_VMEMPTR:[0-9]+]]

// CALLGRAPH-DAG: define{{.*}} i32 @_ZN4Base3getEi(ptr {{.*}}, i32 {{.*}}){{.*}} !callgraph ![[CG_METHOD_TYPE:[0-9]+]]
// CFI-NOCG-DAG: define{{.*}} void @_Z12icall_targeti(i32 {{.*}}){{.*}} !type ![[FN_TYPE:[0-9]+]] !type ![[FN_GEN_TYPE:[0-9]+]]
// CALLGRAPH-DAG: define{{.*}} void @_Z12icall_targeti(i32 {{.*}}){{.*}} !type ![[FN_TYPE:[0-9]+]] !type ![[FN_GEN_TYPE:[0-9]+]]{{.*}} !callgraph ![[CG_FN_TYPE:[0-9]+]]
// KCFI-DAG: define{{.*}} void @_Z12icall_targeti(i32 {{.*}}){{.*}} !kcfi_type ![[KCFI_FN_TYPE:[0-9]+]]

// CHECK-LABEL: define{{.*}} i32 @_Z10test_callsP4BaseMS_FiiEP3BoxIiEPFviEi(
// KCFI-DAG: call i1 @llvm.public.type.test(ptr {{.*}}, metadata !"_ZTS4Base")
// KCFI-DAG: call i1 @llvm.public.type.test(ptr {{.*}}, metadata !"_ZTS3BoxIiE")
// KCFI-DAG: call void %{{.*}}(i32 {{.*}}) [ "kcfi"(i32 1162514891) ]
// CFI-PUBLIC-DAG: call i1 @llvm.public.type.test(ptr {{.*}}, metadata !"_ZTSM4BaseFiiE.virtual")
// CFI-HIDDEN-DAG: call { ptr, i1 } @llvm.type.checked.load(ptr {{.*}}, i32 0, metadata !"_ZTS4Base")
// CFI-HIDDEN-DAG: call i1 @llvm.type.test(ptr {{.*}}, metadata !"_ZTSM4BaseFiiE.virtual")
// CFI-HIDDEN-DAG: call i1 @llvm.type.test(ptr {{.*}}, metadata !"_ZTSM4BaseFiiE")
// CFI-HIDDEN-DAG: call { ptr, i1 } @llvm.type.checked.load(ptr {{.*}}, i32 0, metadata !"_ZTS3BoxIiE")
// CFI-DAG: call i1 @llvm.type.test(ptr {{.*}}, metadata !"_ZTSFvu3i32E.normalized")
// CALLGRAPH-DAG: call noundef i32 %{{.*}}(ptr {{.*}}, i32 {{.*}}), !callee_type ![[CG_METHOD_CALLEE:[0-9]+]]
// CALLGRAPH-DAG: call void %{{.*}}(i32 {{.*}}), !callee_type ![[CG_FN_CALLEE:[0-9]+]]

// CHECK-DAG: ![[BASE_TYPE]] = !{i64 16, !"_ZTS4Base"}
// CHECK-DAG: ![[BASE_VMEMPTR]] = !{i64 16, !"_ZTSM4BaseFiiE.virtual"}
// CHECK-DAG: ![[DERIVED_TYPE]] = !{i64 16, !"_ZTS7Derived"}
// CHECK-DAG: ![[DERIVED_VMEMPTR]] = !{i64 16, !"_ZTSM7DerivedFiiE.virtual"}
// CHECK-DAG: ![[BOX_TYPE]] = !{i64 16, !"_ZTS3BoxIiE"}
// CHECK-DAG: ![[BOX_VMEMPTR]] = !{i64 16, !"_ZTSM3BoxIiEFiiE.virtual"}
// CFI-DAG: ![[FN_TYPE]] = !{i64 0, !"_ZTSFvu3i32E.normalized"}
// CFI-DAG: ![[FN_GEN_TYPE]] = !{i64 0, !"_ZTSFvu3i32E.normalized.generalized"}
// KCFI-DAG: ![[KCFI_FN_TYPE]] = !{i32 1162514891}
// CALLGRAPH-DAG: ![[CG_METHOD_TYPE]] = !{!"_ZTSFiiE"}
// CALLGRAPH-DAG: ![[CG_METHOD_CALLEE]] = !{![[CG_METHOD_TYPE]]}
// CALLGRAPH-DAG: ![[CG_FN_TYPE]] = !{!"_ZTSFviE"}
// CALLGRAPH-DAG: ![[CG_FN_CALLEE]] = !{![[CG_FN_TYPE]]}
