// RUN: %clang_cc1 -std=c++17 -mconstructor-aliases -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// RUN: %clang_cc1 -std=c++17 -mconstructor-aliases -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -clangir-disable-emit-cxx-default %s -o %t-disable.cir
// RUN: FileCheck --input-file=%t-disable.cir %s --check-prefix=DISABLE

int strlen(char const *);

struct String {
  long size;
  long capacity;

  String() : size{0}, capacity{0} {}
  String(char const *s) : size{strlen(s)}, capacity{size} {}
  // StringView::StringView(String const&)
  //
  // CHECK: cir.func linkonce_odr @_ZN10StringViewC2ERK6String
  // CHECK:   %0 = cir.alloca !cir.ptr<!ty_22StringView22>, cir.ptr <!cir.ptr<!ty_22StringView22>>, ["this", init] {alignment = 8 : i64}
  // CHECK:   %1 = cir.alloca !cir.ptr<!ty_22String22>, cir.ptr <!cir.ptr<!ty_22String22>>, ["s", init] {alignment = 8 : i64}
  // CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_22StringView22>
  // CHECK:   cir.store %arg1, %1 : !cir.ptr<!ty_22String22>
  // CHECK:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22StringView22>>

  // Get address of `this->size`

  // CHECK:   %3 = "cir.struct_element_addr"(%2) <{member_index = 0 : index, member_name = "size"}>

  // Get address of `s`

  // CHECK:   %4 = cir.load %1 : cir.ptr <!cir.ptr<!ty_22String22>>

  // Get the address of s.size

  // CHECK:   %5 = "cir.struct_element_addr"(%4) <{member_index = 0 : index, member_name = "size"}>

  // Load value from s.size and store in this->size

  // CHECK:   %6 = cir.load %5 : cir.ptr <!s64i>, !s64i
  // CHECK:   cir.store %6, %3 : !s64i, cir.ptr <!s64i>
  // CHECK:   cir.return
  // CHECK: }

  // DISABLE: cir.func linkonce_odr @_ZN10StringViewC2ERK6String
  // DISABLE-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22StringView22>, cir.ptr <!cir.ptr<!ty_22StringView22>>, ["this", init] {alignment = 8 : i64}

  // StringView::operator=(StringView&&)
  //
  // CHECK: cir.func linkonce_odr @_ZN10StringViewaSEOS_
  // CHECK:   %0 = cir.alloca !cir.ptr<!ty_22StringView22>, cir.ptr <!cir.ptr<!ty_22StringView22>>, ["this", init] {alignment = 8 : i64}
  // CHECK:   %1 = cir.alloca !cir.ptr<!ty_22StringView22>, cir.ptr <!cir.ptr<!ty_22StringView22>>, ["", init] {alignment = 8 : i64}
  // CHECK:   %2 = cir.alloca !cir.ptr<!ty_22StringView22>, cir.ptr <!cir.ptr<!ty_22StringView22>>, ["__retval"] {alignment = 8 : i64}
  // CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_22StringView22>
  // CHECK:   cir.store %arg1, %1 : !cir.ptr<!ty_22StringView22>
  // CHECK:   %3 = cir.load deref %0 : cir.ptr <!cir.ptr<!ty_22StringView22>>
  // CHECK:   %4 = cir.load %1 : cir.ptr <!cir.ptr<!ty_22StringView22>>
  // CHECK:   %5 = "cir.struct_element_addr"(%4) <{member_index = 0 : index, member_name = "size"}>
  // CHECK:   %6 = cir.load %5 : cir.ptr <!s64i>, !s64i
  // CHECK:   %7 = "cir.struct_element_addr"(%3) <{member_index = 0 : index, member_name = "size"}>
  // CHECK:   cir.store %6, %7 : !s64i, cir.ptr <!s64i>
  // CHECK:   cir.store %3, %2 : !cir.ptr<!ty_22StringView22>
  // CHECK:   %8 = cir.load %2 : cir.ptr <!cir.ptr<!ty_22StringView22>>
  // CHECK:   cir.return %8 : !cir.ptr<!ty_22StringView22>
  // CHECK: }

  // DISABLE: cir.func private @_ZN10StringViewaSEOS_
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

// CHECK: cir.func @main() -> !s32i
// CHECK:     %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca !ty_22StringView22, cir.ptr <!ty_22StringView22>, ["sv", init] {alignment = 8 : i64}
// CHECK:     cir.call @_ZN10StringViewC2Ev(%1) : (!cir.ptr<!ty_22StringView22>) -> ()
// CHECK:     cir.scope {
// CHECK:       %3 = cir.alloca !ty_22String22, cir.ptr <!ty_22String22>, ["s", init] {alignment = 8 : i64}
// CHECK:       %4 = cir.get_global @".str" : cir.ptr <!cir.array<!s8i x 3>>
// CHECK:       %5 = cir.cast(array_to_ptrdecay, %4 : !cir.ptr<!cir.array<!s8i x 3>>), !cir.ptr<!s8i>
// CHECK:       cir.call @_ZN6StringC2EPKc(%3, %5) : (!cir.ptr<!ty_22String22>, !cir.ptr<!s8i>) -> ()
// CHECK:       cir.scope {
// CHECK:         %6 = cir.alloca !ty_22StringView22, cir.ptr <!ty_22StringView22>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK:         cir.call @_ZN10StringViewC2ERK6String(%6, %3) : (!cir.ptr<!ty_22StringView22>, !cir.ptr<!ty_22String22>) -> ()
// CHECK:         %7 = cir.call @_ZN10StringViewaSEOS_(%1, %6) : (!cir.ptr<!ty_22StringView22>, !cir.ptr<!ty_22StringView22>) -> !cir.ptr<!ty_22StringView22>
// CHECK:       }
// CHECK:     }
// CHECK:     %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK:     cir.return %2 : !s32i
// CHECK: }
