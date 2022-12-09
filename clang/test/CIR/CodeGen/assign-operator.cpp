// RUN: %clang_cc1 -std=c++17 -mconstructor-aliases -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -mconstructor-aliases -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -clangir-disable-emit-cxx-default %s -o - | FileCheck %s --check-prefix=DISABLE

int strlen(char const *);

struct String {
  long size;
  long capacity;

  String() : size{0}, capacity{0} {}
  String(char const *s) : size{strlen(s)}, capacity{size} {}
  // StringView::StringView(String const&)
  //
  // CHECK: cir.func linkonce_odr @_ZN10StringViewC2ERK6String
  // CHECK:   %0 = cir.alloca !cir.ptr<!ty_22struct2EStringView22>, cir.ptr <!cir.ptr<!ty_22struct2EStringView22>>, ["this", init] {alignment = 8 : i64}
  // CHECK:   %1 = cir.alloca !cir.ptr<!ty_22struct2EString22>, cir.ptr <!cir.ptr<!ty_22struct2EString22>>, ["s", init] {alignment = 8 : i64}
  // CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_22struct2EStringView22>
  // CHECK:   cir.store %arg1, %1 : !cir.ptr<!ty_22struct2EString22>
  // CHECK:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22struct2EStringView22>>
  // CHECK:   %3 = "cir.struct_element_addr"(%0) <{member_name = "size"}>
  // CHECK:   %4 = cir.load %1 : cir.ptr <!cir.ptr<!ty_22struct2EString22>>
  // CHECK:   %5 = "cir.struct_element_addr"(%0) <{member_name = "size"}>
  // CHECK:   %6 = cir.load %5 : cir.ptr <i64>, i64
  // CHECK:   cir.store %6, %3 : i64, cir.ptr <i64>
  // CHECK:   cir.return
  // CHECK: }

  // DISABLE: cir.func linkonce_odr @_ZN10StringViewC2ERK6String
  // DISABLE-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22struct2EStringView22>, cir.ptr <!cir.ptr<!ty_22struct2EStringView22>>, ["this", init] {alignment = 8 : i64}

  // StringView::operator=(StringView&&)
  //
  // CHECK: cir.func linkonce_odr @_ZN10StringViewaSEOS_
  // CHECK:   %0 = cir.alloca !cir.ptr<!ty_22struct2EStringView22>, cir.ptr <!cir.ptr<!ty_22struct2EStringView22>>, ["this", init] {alignment = 8 : i64}
  // CHECK:   %1 = cir.alloca !cir.ptr<!ty_22struct2EStringView22>, cir.ptr <!cir.ptr<!ty_22struct2EStringView22>>, ["", init] {alignment = 8 : i64}
  // CHECK:   %2 = cir.alloca !cir.ptr<!ty_22struct2EStringView22>, cir.ptr <!cir.ptr<!ty_22struct2EStringView22>>, ["__retval"] {alignment = 8 : i64}
  // CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_22struct2EStringView22>
  // CHECK:   cir.store %arg1, %1 : !cir.ptr<!ty_22struct2EStringView22>
  // CHECK:   %3 = cir.load deref %0 : cir.ptr <!cir.ptr<!ty_22struct2EStringView22>>
  // CHECK:   %4 = cir.load %1 : cir.ptr <!cir.ptr<!ty_22struct2EStringView22>>
  // CHECK:   %5 = "cir.struct_element_addr"(%0) <{member_name = "size"}>
  // CHECK:   %6 = cir.load %5 : cir.ptr <i64>, i64
  // CHECK:   %7 = "cir.struct_element_addr"(%0) <{member_name = "size"}>
  // CHECK:   cir.store %6, %7 : i64, cir.ptr <i64>
  // CHECK:   cir.store %3, %2 : !cir.ptr<!ty_22struct2EStringView22>
  // CHECK:   %8 = cir.load %2 : cir.ptr <!cir.ptr<!ty_22struct2EStringView22>>
  // CHECK:   cir.return %8 : !cir.ptr<!ty_22struct2EStringView22>
  // CHECK: }

  // DISABLE: cir.func @_ZN10StringViewaSEOS_
  // DISABLE-NEXT: cir.func @main()
};

struct StringView {
  long size;

  StringView(const String &s) : size{s.size} {}
  StringView() : size{0} {}
};

int main() {
  StringView sv;
  {
    String s = "Hi";
    sv = s;
  }
}

// CHECK: cir.func @main() -> i32 {
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["__retval"] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca !ty_22struct2EStringView22, cir.ptr <!ty_22struct2EStringView22>, ["sv"] {alignment = 8 : i64}
// CHECK:     cir.call @_ZN10StringViewC2Ev(%1) : (!cir.ptr<!ty_22struct2EStringView22>) -> ()
// CHECK:     cir.scope {
// CHECK:       %3 = cir.alloca !ty_22struct2EString22, cir.ptr <!ty_22struct2EString22>, ["s"] {alignment = 8 : i64}
// CHECK:       %4 = cir.alloca !ty_22struct2EStringView22, cir.ptr <!ty_22struct2EStringView22>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK:       %5 = cir.get_global @".str" : cir.ptr <!cir.array<i8 x 3>>
// CHECK:       %6 = cir.cast(array_to_ptrdecay, %5 : !cir.ptr<!cir.array<i8 x 3>>), !cir.ptr<i8>
// CHECK:       cir.call @_ZN6StringC2EPKc(%3, %6) : (!cir.ptr<!ty_22struct2EString22>, !cir.ptr<i8>) -> ()
// CHECK:       cir.call @_ZN10StringViewC2ERK6String(%4, %3) : (!cir.ptr<!ty_22struct2EStringView22>, !cir.ptr<!ty_22struct2EString22>) -> ()
// CHECK:       %7 = cir.call @_ZN10StringViewaSEOS_(%1, %4) : (!cir.ptr<!ty_22struct2EStringView22>, !cir.ptr<!ty_22struct2EStringView22>) -> !cir.ptr<!ty_22struct2EStringView22>
// CHECK:     }
// CHECK:     %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:     cir.return %2 : i32
// CHECK:   }
