// RUN: %clang_cc1 -triple x86_64-gnu-linux -x c++ -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-gnu-linux -x c++ -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-gnu-linux -x c++ -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// All cases from CodeGen/attr-noundef.cpp (x86_64 only).
// Tests noundef placement on structs, unions, this pointers, vectors,
// function/array pointers, member pointers, nullptr_t, and _BitInt.

//************ Passing structs by value

namespace check_structs {
struct Trivial {
  int a;
};
Trivial ret_trivial() { return {}; }
void pass_trivial(Trivial e) {}

// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs11ret_trivialEv
// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs12pass_trivialENS_7TrivialE

// LLVM-LABEL: define {{.*}} @_ZN13check_structs11ret_trivialEv(
// LLVM-LABEL: define {{.*}} void @_ZN13check_structs12pass_trivialENS_7TrivialE(

// OGCG: define{{.*}} i32 @_ZN13check_structs11ret_trivialEv
// OGCG: define{{.*}} void @_ZN13check_structs12pass_trivialENS_7TrivialE{{.*}}(i32 %

struct NoCopy {
  int a;
  NoCopy(NoCopy &) = delete;
};
NoCopy ret_nocopy() { return {}; }
void pass_nocopy(NoCopy e) {}

// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs10ret_nocopyEv
// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs11pass_nocopyENS_6NoCopyE

// LLVM-LABEL: define {{.*}} @_ZN13check_structs10ret_nocopyEv(
// LLVM-LABEL: define {{.*}} void @_ZN13check_structs11pass_nocopyENS_6NoCopyE(

// OGCG: define{{.*}} void @_ZN13check_structs10ret_nocopyEv{{.*}}(ptr dead_on_unwind noalias writable sret({{[^)]+}}) align 4 %
// OGCG: define{{.*}} void @_ZN13check_structs11pass_nocopyENS_6NoCopyE{{.*}}(ptr noundef dead_on_return %

struct Huge {
  int a[1024];
};
Huge ret_huge() { return {}; }
void pass_huge(Huge h) {}

// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs8ret_hugeEv
// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs9pass_hugeENS_4HugeE

// LLVM-LABEL: define {{.*}} @_ZN13check_structs8ret_hugeEv(
// LLVM-LABEL: define {{.*}} void @_ZN13check_structs9pass_hugeENS_4HugeE(

// OGCG: define{{.*}} void @_ZN13check_structs8ret_hugeEv{{.*}}(ptr dead_on_unwind noalias writable sret({{[^)]+}}) align 4 %
// OGCG: define{{.*}} void @_ZN13check_structs9pass_hugeENS_4HugeE{{.*}}(ptr noundef
} // namespace check_structs

//************ Passing unions by value

namespace check_unions {
union Trivial {
  int a;
};
Trivial ret_trivial() { return {}; }
void pass_trivial(Trivial e) {}

// CIR-LABEL: cir.func {{.*}} @_ZN12check_unions11ret_trivialEv
// CIR-LABEL: cir.func {{.*}} @_ZN12check_unions12pass_trivialENS_7TrivialE

// LLVM-LABEL: define {{.*}} @_ZN12check_unions11ret_trivialEv(
// LLVM-LABEL: define {{.*}} void @_ZN12check_unions12pass_trivialENS_7TrivialE(

// OGCG: define{{.*}} i32 @_ZN12check_unions11ret_trivialEv
// OGCG: define{{.*}} void @_ZN12check_unions12pass_trivialENS_7TrivialE{{.*}}(i32 %

union NoCopy {
  int a;
  NoCopy(NoCopy &) = delete;
};
NoCopy ret_nocopy() { return {}; }
void pass_nocopy(NoCopy e) {}

// CIR-LABEL: cir.func {{.*}} @_ZN12check_unions10ret_nocopyEv
// CIR-LABEL: cir.func {{.*}} @_ZN12check_unions11pass_nocopyENS_6NoCopyE

// LLVM-LABEL: define {{.*}} @_ZN12check_unions10ret_nocopyEv(
// LLVM-LABEL: define {{.*}} void @_ZN12check_unions11pass_nocopyENS_6NoCopyE(

// OGCG: define{{.*}} void @_ZN12check_unions10ret_nocopyEv{{.*}}(ptr dead_on_unwind noalias writable sret({{[^)]+}}) align 4 %
// OGCG: define{{.*}} void @_ZN12check_unions11pass_nocopyENS_6NoCopyE{{.*}}(ptr noundef dead_on_return %
} // namespace check_unions

//************ Passing `this` pointers

namespace check_this {
struct Object {
  int data[];

  Object() {
    this->data[0] = 0;
  }
  int getData() {
    return this->data[0];
  }
  Object *getThis() {
    return this;
  }
};

void use_object() {
  Object obj;
  obj.getData();
  obj.getThis();
}

// CIR-LABEL: cir.func {{.*}} @_ZN10check_this10use_objectEv
// CIR:   cir.call @_ZN10check_this6ObjectC1Ev
// CIR:   cir.call @_ZN10check_this6Object7getDataEv
// CIR:   cir.call @_ZN10check_this6Object7getThisEv

// this pointer: noundef nonnull dereferenceable align
// LLVM: define linkonce_odr void @_ZN10check_this6Object{{.*}}(ptr noundef nonnull align 4 dereferenceable(1) %
// LLVM: define linkonce_odr noundef i32 @_ZN10check_this6Object7getDataEv(ptr noundef nonnull align 4 dereferenceable(1) %
// LLVM: define linkonce_odr noundef ptr @_ZN10check_this6Object7getThisEv(ptr noundef nonnull align 4 dereferenceable(1) %

// OGCG: define linkonce_odr void @_ZN10check_this6ObjectC1Ev(ptr noundef nonnull align 4 dereferenceable(1) %
// OGCG: define linkonce_odr noundef i32 @_ZN10check_this6Object7getDataEv(ptr noundef nonnull align 4 dereferenceable(1) %
// OGCG: define linkonce_odr noundef ptr @_ZN10check_this6Object7getThisEv(ptr noundef nonnull align 4 dereferenceable(1) %
} // namespace check_this

//************ Passing vector types

namespace check_vecs {
typedef int __attribute__((vector_size(12))) i32x3;
i32x3 ret_vec() {
  return {};
}
void pass_vec(i32x3 v) {
}

// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs7ret_vecEv
// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs8pass_vecEDv3_i

// LLVM: define {{.*}} noundef <3 x i32> @_ZN10check_vecs7ret_vecEv(
// LLVM: define {{.*}} void @_ZN10check_vecs8pass_vecEDv3_i(<3 x i32> noundef %

// OGCG: define {{.*}} noundef <3 x i32> @_ZN10check_vecs7ret_vecEv(
// OGCG: define {{.*}} void @_ZN10check_vecs8pass_vecEDv3_i(<3 x i32> noundef %
} // namespace check_vecs

//************ Passing exotic types

namespace check_exotic {
struct Object {
  int mfunc();
  int mdata;
};
typedef int Object::*mdptr;
typedef int (Object::*mfptr)();
typedef decltype(nullptr) nullptr_t;
typedef int (*arrptr)[32];
typedef int (*fnptr)(int);

arrptr ret_arrptr() {
  return nullptr;
}
fnptr ret_fnptr() {
  return nullptr;
}
mdptr ret_mdptr() {
  return nullptr;
}
mfptr ret_mfptr() {
  return nullptr;
}
nullptr_t ret_npt() {
  return nullptr;
}
void pass_npt(nullptr_t t) {
}
_BitInt(3) ret_BitInt() {
  return 0;
}
void pass_BitInt(_BitInt(3) e) {
}
void pass_large_BitInt(_BitInt(127) e) {
}

// Pointers to arrays/functions: always noundef
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic10ret_arrptrEv
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic9ret_fnptrEv

// LLVM: define {{.*}} noundef ptr @_ZN12check_exotic10ret_arrptrEv(
// LLVM: define {{.*}} noundef ptr @_ZN12check_exotic9ret_fnptrEv(

// OGCG: define {{.*}} noundef ptr @_ZN12check_exotic10ret_arrptrEv(
// OGCG: define {{.*}} noundef ptr @_ZN12check_exotic9ret_fnptrEv(

// Member pointers: never noundef
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic9ret_mdptrEv
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic9ret_mfptrEv

// LLVM: define {{.*}} i64 @_ZN12check_exotic9ret_mdptrEv(
// LLVM: define {{.*}} { i64, i64 } @_ZN12check_exotic9ret_mfptrEv(

// OGCG: define {{.*}} i64 @_ZN12check_exotic9ret_mdptrEv(
// OGCG: define {{.*}} { i64, i64 } @_ZN12check_exotic9ret_mfptrEv(

// nullptr_t: never noundef
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic7ret_nptEv
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic8pass_nptEDn

// LLVM: define {{.*}} ptr @_ZN12check_exotic7ret_nptEv(
// LLVM: define {{.*}} void @_ZN12check_exotic8pass_nptEDn(ptr %

// OGCG: define {{.*}} ptr @_ZN12check_exotic7ret_nptEv(
// OGCG: define {{.*}} void @_ZN12check_exotic8pass_nptEDn(ptr %

// _BitInt types
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic10ret_BitIntEv
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic11pass_BitIntEDB3_
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic17pass_large_BitIntEDB127_

// LLVM: define {{.*}} i3 @_ZN12check_exotic10ret_BitIntEv(
// LLVM: define {{.*}} void @_ZN12check_exotic11pass_BitIntEDB3_(i3 %
// LLVM: define {{.*}} void @_ZN12check_exotic17pass_large_BitIntEDB127_(i127 %

// OGCG: define {{.*}} noundef signext i3 @_ZN12check_exotic10ret_BitIntEv(
// OGCG: define {{.*}} void @_ZN12check_exotic11pass_BitIntEDB3_(i3 noundef signext %
// OGCG: define {{.*}} void @_ZN12check_exotic17pass_large_BitIntEDB127_(i64 noundef %{{.*}}, i64 noundef %
} // namespace check_exotic
