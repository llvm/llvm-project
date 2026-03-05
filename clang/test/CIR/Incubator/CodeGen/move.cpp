// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

namespace std {

template<typename T> struct remove_reference       { typedef T type; };
template<typename T> struct remove_reference<T &>  { typedef T type; };
template<typename T> struct remove_reference<T &&> { typedef T type; };

template<typename T>
typename remove_reference<T>::type &&move(T &&t) noexcept;

struct string {
  string();
};

} // std namespace

// CHECK: ![[StdString:rec_.*]] = !cir.record<struct "std::string" padded {!u8i}>

std::string getstr();
void emplace(std::string &&s);

void t() {
  emplace(std::move(getstr()));
}

// FIXME: we should explicitly model std::move here since it will
// be useful at least for the lifetime checker.

// CHECK: cir.func {{.*}} @_Z1tv()
// CHECK:   %[[#Addr:]] = cir.alloca ![[StdString]], {{.*}} ["ref.tmp0"]
// CHECK:   %[[#RValStr:]] = cir.call @_Z6getstrv() : () -> ![[StdString]]
// CHECK:   cir.store{{.*}} %[[#RValStr]], %[[#Addr]]
// CHECK:   cir.call @_Z7emplaceOSt6string(%[[#Addr]])
// CHECK:   cir.return
// CHECK: }

struct S {
  S() = default;
  S(S&&) = default;

  int val;
};

// CHECK-LABEL:   cir.func {{.*}} @_ZN1SC1EOS_
// CHECK-SAME:      special_member<#cir.cxx_ctor<!rec_S, move>>

void test_ctor() {
// CHECK-LABEL:   cir.func {{.*}} @_Z9test_ctorv()
// CHECK:           %[[VAR_A:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>
// CHECK:           %[[VAR_B:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>
// CHECK:           cir.call @_ZN1SC1EOS_(%[[VAR_B]], %[[VAR_A]]) : (!cir.ptr<!rec_S>, !cir.ptr<!rec_S>) -> ()
// CHECK:           cir.return
// CHECK:         }

  S a;
  S b(std::move(a));
}
