// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
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

// CHECK: ![[StdString:ty_.*]] = !cir.struct<struct "std::string" {!u8i}>

std::string getstr();
void emplace(std::string &&s);

void t() {
  emplace(std::move(getstr()));
}

// FIXME: we should explicitly model std::move here since it will
// be useful at least for the lifetime checker.

// CHECK: cir.func @_Z1tv()
// CHECK:   %[[#Addr:]] = cir.alloca ![[StdString]], {{.*}} ["ref.tmp0"]
// CHECK:   %[[#RValStr:]] = cir.call @_Z6getstrv() : () -> ![[StdString]]
// CHECK:   cir.store %[[#RValStr]], %[[#Addr]]
// CHECK:   cir.call @_Z7emplaceOSt6string(%[[#Addr]])
// CHECK:   cir.return
// CHECK: }
