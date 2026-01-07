// RUN: %check_clang_tidy -check-suffixes=,DEFAULT -std=c++17-or-later %s bugprone-invalid-enum-default-initialization %t
// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-invalid-enum-default-initialization %t -- -config="{CheckOptions: {bugprone-invalid-enum-default-initialization.IgnoredEnums: '::MyEnum'}}"

enum class Enum0: int {
  A = 0,
  B
};

enum class Enum1: int {
  A = 1,
  B
};

enum Enum2 {
  Enum_A = 4,
  Enum_B
};

Enum0 E0_1{};
Enum0 E0_2 = Enum0();
Enum0 E0_3;
Enum0 E0_4{0};
Enum0 E0_5{Enum0::A};
Enum0 E0_6{Enum0::B};

Enum1 E1_1{};
// CHECK-NOTES: :[[@LINE-1]]:11: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :9:12: note: enum is defined here
Enum1 E1_2 = Enum1();
// CHECK-NOTES: :[[@LINE-1]]:14: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :9:12: note: enum is defined here
Enum1 E1_3;
Enum1 E1_4{0};
Enum1 E1_5{Enum1::A};
Enum1 E1_6{Enum1::B};

Enum2 E2_1{};
// CHECK-NOTES: :[[@LINE-1]]:11: warning: enum value of type 'Enum2' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :14:6: note: enum is defined here
Enum2 E2_2 = Enum2();
// CHECK-NOTES: :[[@LINE-1]]:14: warning: enum value of type 'Enum2' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :14:6: note: enum is defined here

void f1() {
  static Enum1 S; // FIMXE: warn for this?
  Enum1 A;
  Enum1 B = Enum1();
  // CHECK-NOTES: :[[@LINE-1]]:13: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
  // CHECK-NOTES: :9:12: note: enum is defined here
  int C = int();
}

void f2() {
  Enum1 A{};
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
  // CHECK-NOTES: :9:12: note: enum is defined here
  Enum1 B = Enum1();
  // CHECK-NOTES: :[[@LINE-1]]:13: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
  // CHECK-NOTES: :9:12: note: enum is defined here
  Enum1 C[5] = {{}};
  // CHECK-NOTES: :[[@LINE-1]]:16: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
  // CHECK-NOTES: :9:12: note: enum is defined here
  // CHECK-NOTES: :[[@LINE-3]]:17: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
  // CHECK-NOTES: :9:12: note: enum is defined here
  Enum1 D[5] = {}; // FIMXE: warn for this?
  // CHECK-NOTES: :[[@LINE-1]]:16: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
  // CHECK-NOTES: :9:12: note: enum is defined here
}

struct S1 {
  Enum1 E_1{};
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
  // CHECK-NOTES: :9:12: note: enum is defined here
  Enum1 E_2 = Enum1();
  // CHECK-NOTES: :[[@LINE-1]]:15: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
  // CHECK-NOTES: :9:12: note: enum is defined here
  Enum1 E_3;
  Enum1 E_4;
  Enum1 E_5;

  S1() :
    E_3{},
    // CHECK-NOTES: :[[@LINE-1]]:8: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
    // CHECK-NOTES: :9:12: note: enum is defined here
    E_4(),
    // CHECK-NOTES: :[[@LINE-1]]:8: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
    // CHECK-NOTES: :9:12: note: enum is defined here
    E_5{Enum1::B}
  {}
};

struct S2 {
  Enum0 X;
  Enum1 Y;
  Enum2 Z;
};

struct S3 {
  S2 X;
  int Y;
};

struct S4 : public S3 {
  int Z;
};

struct S5 {
  S2 X[3];
  int Y;
};

S2 VarS2{};
// CHECK-NOTES: :[[@LINE-1]]:9: warning: enum value of type 'Enum1' initialized with invalid value of 0
// CHECK-NOTES: :9:12: note: enum is defined here
// CHECK-NOTES: :[[@LINE-3]]:9: warning: enum value of type 'Enum2' initialized with invalid value of 0
// CHECK-NOTES: :14:6: note: enum is defined here
S3 VarS3{};
// CHECK-NOTES: :[[@LINE-1]]:10: warning: enum value of type 'Enum1' initialized with invalid value of 0
// CHECK-NOTES: :9:12: note: enum is defined here
// CHECK-NOTES: :[[@LINE-3]]:10: warning: enum value of type 'Enum2' initialized with invalid value of 0
// CHECK-NOTES: :14:6: note: enum is defined here
S4 VarS4{};
// CHECK-NOTES: :[[@LINE-1]]:10: warning: enum value of type 'Enum1' initialized with invalid value of 0
// CHECK-NOTES: :9:12: note: enum is defined here
// CHECK-NOTES: :[[@LINE-3]]:10: warning: enum value of type 'Enum2' initialized with invalid value of 0
// CHECK-NOTES: :14:6: note: enum is defined here
S5 VarS5{};
// CHECK-NOTES: :[[@LINE-1]]:10: warning: enum value of type 'Enum1' initialized with invalid value of 0
// CHECK-NOTES: :9:12: note: enum is defined here

enum class EnumFwd;

EnumFwd Fwd{};

enum class EnumEmpty {};

EnumEmpty Empty{};

template<typename T>
struct Templ {
  T Mem1{};
  // CHECK-NOTES: :[[@LINE-1]]:9: warning: enum value of type 'Enum1' initialized with invalid value of 0
  // CHECK-NOTES: :9:12: note: enum is defined here
};

Templ<Enum1> TemplVar;

enum MyEnum {
  A = 1,
  B
};

MyEnum MyEnumVar{};
// CHECK-NOTES-DEFAULT: :[[@LINE-1]]:17: warning: enum value of type 'MyEnum' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES-DEFAULT: :148:6: note: enum is defined here

namespace std {
  enum errc {
    A = 1,
    B
  };
}

std::errc err{};
