// RUN: %clang_cc1 -triple x86_64-gnu-linux -x c++ -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-gnu-linux -x c++ -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-gnu-linux -x c++ -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

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

// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs11ret_trivialEv() -> !s32i
// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs12pass_trivialENS_7TrivialE(%arg0: !s32i loc(

// LLVM: define dso_local i32 @_ZN13check_structs11ret_trivialEv()
// LLVM: define dso_local void @_ZN13check_structs12pass_trivialENS_7TrivialE(i32 %

struct NoCopy {
  int a;
  NoCopy(NoCopy &) = delete;
};
NoCopy ret_nocopy() { return {}; }
void pass_nocopy(NoCopy e) {}

// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs10ret_nocopyEv
// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs11pass_nocopyENS_6NoCopyE(%arg0: !cir.ptr<!rec_check_structs3A3ANoCopy> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}

// LLVM: define dso_local void @_ZN13check_structs10ret_nocopyEv(ptr dead_on_unwind noalias writable sret(%"struct.check_structs::NoCopy") align 4 %
// LLVMCIR: define dso_local void @_ZN13check_structs11pass_nocopyENS_6NoCopyE(ptr nofreeobj noundef align 4 dereferenceable(4) %
// OGCG: define dso_local void @_ZN13check_structs11pass_nocopyENS_6NoCopyE(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %

struct Huge {
  int a[1024];
};
Huge ret_huge() { return {}; }
void pass_huge(Huge h) {}

// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs8ret_hugeEv
// CIR-LABEL: cir.func {{.*}} @_ZN13check_structs9pass_hugeENS_4HugeE(%arg0: !cir.ptr<!rec_check_structs3A3AHuge> {llvm.align = 8 : i64, llvm.byval = !rec_check_structs3A3AHuge, llvm.noundef}

// LLVM: define dso_local void @_ZN13check_structs8ret_hugeEv(ptr dead_on_unwind noalias writable sret(%"struct.check_structs::Huge") align 4 %
// LLVM: define dso_local void @_ZN13check_structs9pass_hugeENS_4HugeE(ptr noundef byval(%"struct.check_structs::Huge") align 8 %
} // namespace check_structs

//************ Passing unions by value

namespace check_unions {
union Trivial {
  int a;
};
Trivial ret_trivial() { return {}; }
void pass_trivial(Trivial e) {}

// CIR-LABEL: cir.func {{.*}} @_ZN12check_unions11ret_trivialEv() -> !s32i
// CIR-LABEL: cir.func {{.*}} @_ZN12check_unions12pass_trivialENS_7TrivialE(%arg0: !s32i loc(

// LLVM: define dso_local i32 @_ZN12check_unions11ret_trivialEv()
// LLVM: define dso_local void @_ZN12check_unions12pass_trivialENS_7TrivialE(i32 %

union NoCopy {
  int a;
  NoCopy(NoCopy &) = delete;
};
NoCopy ret_nocopy() { return {}; }
void pass_nocopy(NoCopy e) {}

// CIR-LABEL: cir.func {{.*}} @_ZN12check_unions10ret_nocopyEv
// CIR-LABEL: cir.func {{.*}} @_ZN12check_unions11pass_nocopyENS_6NoCopyE(%arg0: !cir.ptr<!rec_check_unions3A3ANoCopy> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}

// LLVM: define dso_local void @_ZN12check_unions10ret_nocopyEv(ptr dead_on_unwind noalias writable sret(%"union.check_unions::NoCopy") align 4 %
// LLVMCIR: define dso_local void @_ZN12check_unions11pass_nocopyENS_6NoCopyE(ptr nofreeobj noundef align 4 dereferenceable(4) %
// OGCG: define dso_local void @_ZN12check_unions11pass_nocopyENS_6NoCopyE(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %
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
// LLVM: define linkonce_odr void @_ZN10check_this6ObjectC1Ev(ptr noundef nonnull align 4 dereferenceable(1) %
// LLVM: define linkonce_odr noundef i32 @_ZN10check_this6Object7getDataEv(ptr noundef nonnull align 4 dereferenceable(1) %
// LLVM: define linkonce_odr noundef ptr @_ZN10check_this6Object7getThisEv(ptr noundef nonnull align 4 dereferenceable(1) %
} // namespace check_this

//************ Passing vector types

namespace check_vecs {
typedef int __attribute__((vector_size(12))) i32x3;
i32x3 ret_vec() {
  return {};
}
void pass_vec(i32x3 v) {
}
typedef char i8x3 __attribute__((ext_vector_type(3)));
typedef short i16x3 __attribute__((ext_vector_type(3)));
i8x3 ret_i8x3() {
  return {};
}
i16x3 ret_i16x3() {
  return {};
}
void pass_i8x3(i8x3 v) {
}
i8x3 ext_ret_i8x3();
void call_ret_i8x3() {
  ext_ret_i8x3();
}
typedef char i8x4 __attribute__((ext_vector_type(4)));
i8x4 ret_i8x4() {
  return {};
}
i8x4 ext_ret_i8x4();
void call_ret_i8x4() {
  ext_ret_i8x4();
}

// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs7ret_vecEv() -> (!cir.vector<3 x !s32i> {llvm.noundef})
// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs8pass_vecEDv3_i(%arg0: !cir.vector<3 x !s32i> {llvm.noundef}

// LLVM: define dso_local noundef <3 x i32> @_ZN10check_vecs7ret_vecEv()
// LLVM: define dso_local void @_ZN10check_vecs8pass_vecEDv3_i(<3 x i32> noundef %

// Coerced to a wider type: never noundef
// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs8ret_i8x3Ev() -> !u32i
// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs9ret_i16x3Ev() -> !cir.double
// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs9pass_i8x3EDv3_c(%arg0: !u32i loc(
// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs13call_ret_i8x3Ev()
// CIR: cir.call @_ZN10check_vecs12ext_ret_i8x3Ev() : () -> !u32i loc(

// LLVM: define dso_local i32 @_ZN10check_vecs8ret_i8x3Ev()
// LLVM: define dso_local double @_ZN10check_vecs9ret_i16x3Ev()
// LLVM: define dso_local void @_ZN10check_vecs9pass_i8x3EDv3_c(i32 %
// LLVM: define dso_local void @_ZN10check_vecs13call_ret_i8x3Ev()
// LLVM: call i32 @_ZN10check_vecs12ext_ret_i8x3Ev()
// LLVM: declare i32 @_ZN10check_vecs12ext_ret_i8x3Ev()

// Coerced to a type of the same width: noundef
// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs8ret_i8x4Ev() -> (!u32i {llvm.noundef})
// CIR-LABEL: cir.func {{.*}} @_ZN10check_vecs13call_ret_i8x4Ev()
// CIR: cir.call @_ZN10check_vecs12ext_ret_i8x4Ev() : () -> (!u32i {llvm.noundef})

// LLVM: define dso_local noundef i32 @_ZN10check_vecs8ret_i8x4Ev()
// LLVM: define dso_local void @_ZN10check_vecs13call_ret_i8x4Ev()
// LLVM: call noundef i32 @_ZN10check_vecs12ext_ret_i8x4Ev()
// LLVM: declare noundef i32 @_ZN10check_vecs12ext_ret_i8x4Ev()
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

// LLVM: define dso_local noundef ptr @_ZN12check_exotic10ret_arrptrEv()
// LLVM: define dso_local noundef ptr @_ZN12check_exotic9ret_fnptrEv()

// Member pointers: never noundef
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic9ret_mdptrEv
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic9ret_mfptrEv

// LLVM: define dso_local i64 @_ZN12check_exotic9ret_mdptrEv()
// LLVM: define dso_local { i64, i64 } @_ZN12check_exotic9ret_mfptrEv()

// nullptr_t: never noundef
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic7ret_nptEv() -> !cir.ptr<!void>
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic8pass_nptEDn(%arg0: !cir.ptr<!void> loc(

// LLVM: define dso_local ptr @_ZN12check_exotic7ret_nptEv()
// LLVM: define dso_local void @_ZN12check_exotic8pass_nptEDn(ptr %

// _BitInt types
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic10ret_BitIntEv
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic11pass_BitIntEDB3_
// CIR-LABEL: cir.func {{.*}} @_ZN12check_exotic17pass_large_BitIntEDB127_(%arg0: !u64i {llvm.noundef} loc({{.+}}), %arg1: !u64i {llvm.noundef} loc(

// LLVM: define dso_local noundef signext i3 @_ZN12check_exotic10ret_BitIntEv()
// LLVM: define dso_local void @_ZN12check_exotic11pass_BitIntEDB3_(i3 noundef signext %
// LLVM: define dso_local void @_ZN12check_exotic17pass_large_BitIntEDB127_(i64 noundef %{{.+}}, i64 noundef %
} // namespace check_exotic
