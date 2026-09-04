// RUN: %check_clang_tidy %s bugprone-invalid-enum-default-initialization %t

enum Enum1 {
  Enum1_A = 1,
  Enum1_B
};

struct Struct1 {
  int a;
  enum Enum1 b;
};

struct Struct2 {
  struct Struct1 a;
  char b;
};

enum Enum1 E1 = {};
// CHECK-NOTES: :[[@LINE-1]]:17: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :3:6: note: enum is defined here
enum Enum1 E2[10] = {};
// CHECK-NOTES: :[[@LINE-1]]:21: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :3:6: note: enum is defined here
enum Enum1 E3[10] = {Enum1_A};
// CHECK-NOTES: :[[@LINE-1]]:21: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :3:6: note: enum is defined here
enum Enum1 E4[2][2] = {{Enum1_A}, {Enum1_A}};
// CHECK-NOTES: :[[@LINE-1]]:24: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :3:6: note: enum is defined here
// CHECK-NOTES: :[[@LINE-3]]:35: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :3:6: note: enum is defined here
enum Enum1 E5[2][2] = {{Enum1_A, Enum1_A}};
// CHECK-NOTES: :[[@LINE-1]]:23: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :3:6: note: enum is defined here


struct Struct1 S1[2][2] = {{{1, Enum1_A}, {2, Enum1_A}}};
// CHECK-NOTES: :[[@LINE-1]]:27: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :3:6: note: enum is defined here

struct Struct2 S2[3] = {{1}};
// CHECK-NOTES: :[[@LINE-1]]:24: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :3:6: note: enum is defined here
// CHECK-NOTES: :[[@LINE-3]]:26: warning: enum value of type 'Enum1' initialized with invalid value of 0, enum doesn't have a zero-value enumerator
// CHECK-NOTES: :3:6: note: enum is defined here

union Union1 {
  enum Enum1 a;
  int b;
};

// no warnings for union
union Union1 U1 = {};
union Union1 U2[3] = {};
