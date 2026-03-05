// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-move-opt -emit-cir -clangir-verify-diagnostics -mmlir --mlir-print-ir-after=cir-move-opt %s -o /dev/null 2>&1 | FileCheck %s

namespace std {

template<typename T> struct remove_reference       { typedef T type; };
template<typename T> struct remove_reference<T &>  { typedef T type; };
template<typename T> struct remove_reference<T &&> { typedef T type; };

template<typename T>
typename remove_reference<T>::type &&move(T &&t) noexcept;

} // std namespace

struct A {
  A() = default;
  A(A &&) = default;
  A(const A &) = default;
  // expected-remark@above {{move opt: copied object may be unknown}}
  A &operator=(A &&) = default;
  A &operator=(const A &) = default;

  int i;
};

void test_ctor() {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z9test_ctorv()
  // CHECK:           %[[SRC:.*]] = cir.alloca !rec_A
  // CHECK:           %[[DST:.*]] = cir.alloca !rec_A
  // CHECK:           cir.call @_ZN1AC2EOS_(%[[DST]], %[[SRC]])
  // CHECK:           cir.return
  // CHECK:         }

  A a;
  A b(a); // expected-remark {{move opt: transformed copy into move}}
}

void test_assign() {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z11test_assignv()
  // CHECK:           %[[SRC:.*]] = cir.alloca !rec_A
  // CHECK:           %[[DST:.*]] = cir.alloca !rec_A
  // CHECK:           %[[RES:.*]] = cir.call @_ZN1AaSEOS_(%[[DST]], %[[SRC]])
  // CHECK:           cir.return
  // CHECK:         }

  A a, b;
  b = a; // expected-remark {{move opt: transformed copy into move}}
}

void test_may_be_unknown(A *unknown) {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z19test_may_be_unknownP1A(
  // CHECK:           cir.call @_ZN1AaSERKS_(%{{.*}}, %{{.*}})
  // CHECK:         }

  A a, b;
  A *maybe_a = unknown ? unknown : &a;
  b = *maybe_a; // expected-remark {{move opt: copied object may be unknown}}
}

void test_escapes_by_ptr(A **han) {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z19test_escapes_by_ptrPP1A(
  // CHECK:           cir.call @_ZN1AaSERKS_(%{{.*}}, %{{.*}})
  // CHECK:         }

  A a, b;
  *han = &a;
  b = a; // expected-remark {{move opt: copied object may have escaped}}
}

void escape_hatch(A *);
void test_escapes_by_call() {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z20test_escapes_by_callv()
  // CHECK:           cir.call @_ZN1AaSERKS_(%{{.*}}, %{{.*}})
  // CHECK:         }

  A a, b;
  escape_hatch(&a);
  b = a; // expected-remark {{move opt: copied object may have escaped}}
}

int test_live_after_use() {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z19test_live_after_usev()
  // CHECK:           %[[VAL_3:.*]] = cir.call @_ZN1AaSERKS_(%{{.*}}, %{{.*}})
  // CHECK:         }

  A a, b;
  b = a; // expected-remark {{move opt: copied object is alive after use}}
  a.i = 10;
  return a.i;
}

void test_move_after_copy_ctor() {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z25test_move_after_copy_ctorv()
  // CHECK:           %[[VAR_A:.*]] = cir.alloca !rec_A
  // CHECK:           %[[VAR_B:.*]] = cir.alloca !rec_A
  // CHECK:           %[[VAR_C:.*]] = cir.alloca !rec_A
  // CHECK:           cir.call @_ZN1AC1ERKS_(%[[VAR_B]], %[[VAR_A]])
  // CHECK:           cir.call @_ZN1AC1EOS_(%[[VAR_C]], %[[VAR_A]])
  // CHECK:         }

  A a;
  A b(a); // expected-remark {{move opt: copied object is alive after use}}
  A c(std::move(a));
}

void test_move_after_copy_assign() {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z27test_move_after_copy_assignv()
  // CHECK:           %[[VAR_A:.*]] = cir.alloca !rec_A
  // CHECK:           %[[VAR_B:.*]] = cir.alloca !rec_A
  // CHECK:           %[[VAR_C:.*]] = cir.alloca !rec_A
  // CHECK:           %[[VAL_3:.*]] = cir.call @_ZN1AaSERKS_(%[[VAR_B]], %[[VAR_A]])
  // CHECK:           %[[VAL_4:.*]] = cir.call @_ZN1AaSEOS_(%[[VAR_C]], %[[VAR_A]])
  // CHECK:           cir.return
  // CHECK:         }

  A a, b, c;
  b = a; // expected-remark {{move opt: copied object is alive after use}}
  c = std::move(a);
}

void all_effects(A *, A *);
void test_live_after_use_call(A *unknown) {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z24test_live_after_use_callP1A(
  // CHECK:           cir.call @_ZN1AaSERKS_(
  // CHECK:         }

  A a, b;
  b = a; // expected-remark {{move opt: copied object may have escaped}}
  all_effects(&a, unknown);
}

__attribute__((pure))
int read_effect(A *, A *);
void test_live_after_use_pure_call(A *unknown) {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z29test_live_after_use_pure_callP1A(
  // CHECK:           cir.call @_ZN1AaSERKS_(
  // CHECK:         }

  A a, b;
  b = a; // expected-remark {{move opt: copied object is alive after use}}
  read_effect(&a, unknown);
}

__attribute__((const))
int no_effect(A *, A *);
void test_live_after_use_const_call(A *unknown) {
  // CHECK-LABEL:   cir.func {{.*}}dso_local @_Z30test_live_after_use_const_callP1A(
  // CHECK:           cir.call @_ZN1AaSEOS_(
  // CHECK:         }

  A a, b;
  b = a; // expected-remark {{move opt: transformed copy into move}}
  no_effect(&a, unknown);
}

void test_alias(bool cond) {
  A a, b, c;
  A *ptr = cond ? &a : &b;
  c = *ptr; // expected-remark {{move opt: transformed copy into move}}
}
