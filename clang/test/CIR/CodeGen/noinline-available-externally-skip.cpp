// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O0 -disable-llvm-passes -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR,CIRO0

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O0 -disable-llvm-passes -emit-llvm  %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM,LLVMO0

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -disable-llvm-passes -emit-llvm  %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM,LLVMO0

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O2 -disable-llvm-passes -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR,CIRO2

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O2 -disable-llvm-passes -emit-llvm  %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM,LLVMO2

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes -emit-llvm  %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM,LLVMO2

template <typename T>
class Holder {
  T val;
public:
  explicit Holder(T v) : val(v) {}
  // NOT defined.
  // CIR-DAG: cir.func private @_ZNK6HolderIiE4dumpEv{{.*}}attributes {{{.*}}}{{[^{]*}}{{$}}
  // LLVM-DAG: declare void @_ZNK6HolderIiE4dumpEv
  __attribute__((noinline)) void dump() const {}

  // ONLY defined in O2:
  // CIRO0-DAG: cir.func private @_ZNK6HolderIiE4showEv{{.*}}attributes {{{.*}}}{{[^{]*}}{{$}}
  // CIRO2-DAG: cir.func {{.*}}available_externally @_ZNK6HolderIiE4showEv{{.*}}attributes {{{.*}}} {
  // LLVMO0-DAG: declare void @_ZNK6HolderIiE4showEv
  // LLVMO2-DAG: define available_externally void @_ZNK6HolderIiE4showEv
  void show() const {}

  // CIR-DAG: cir.func always_inline {{.*}}available_externally @_ZNK6HolderIiE18dump_always_inlineEv{{.*}}attributes {{{.*}}} {
  // LLVM-DAG: define available_externally void @_ZNK6HolderIiE18dump_always_inlineEv
  __attribute__((always_inline)) void dump_always_inline() const {}
};

// Suppresses instantiation in this TU; dump() and show() are available_externally.
extern template class Holder<int>;

// Normal Definition (wildcard is no_inline, added in O0)
// CIR-DAG: cir.func {{.*}}dso_local @_Z6callerP6HolderIiE{{.*}} attributes {{{.*}}} {
// LLVM-DAG: define {{.*}}dso_local void @_Z6callerP6HolderIiE
void caller(Holder<int> *h) {
  h->dump_always_inline();
  h->dump();
  h->show();
}
