// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -fdump-record-layouts -std=c++17 %s -o %t | FileCheck %s

// CHECK-LABEL:          0 | class Empty (empty)
// CHECK-NEXT:             | [sizeof=1, dsize=1, align=1,
// CHECK-NEXT:             |  nvsize=1, nvalign=1]
// CHECK-LABEL:          0 | class Second
// CHECK-NEXT:           0 |   class Empty (base) (empty)
// CHECK-NEXT:       0:0-0 |   short A
// CHECK-NEXT:             | [sizeof=2, dsize=1, align=2,
// CHECK-NEXT:             |  nvsize=1, nvalign=2]
// CHECK-LABEL:          0 | class Foo
// CHECK-NEXT:           0 |   class Empty (base) (empty)
// CHECK-NEXT:           2 |   class Second NZNoUnique
// CHECK-NEXT:           2 |     class Empty (base) (empty)
// CHECK-NEXT:       2:0-0 |     short A
// CHECK-NEXT:           3 |   char B
// CHECK-NEXT:             | [sizeof=4, dsize=4, align=2,
// CHECK-NEXT:             |  nvsize=4, nvalign=2]

class Empty {};

// CHECK-LABEL: LLVMType:%class.Second = type { i8, i8 }
// CHECK-NEXT:  NonVirtualBaseLLVMType:%class.Second.base = type { i8 }
class Second : Empty {
  short A : 1;
};

// CHECK-LABEL:   LLVMType:%class.Foo = type { [2 x i8], %class.Second.base, i8 }
// CHECK-NEXT:    NonVirtualBaseLLVMType:%class.Foo = type { [2 x i8], %class.Second.base, i8 }
class Foo : Empty {
  [[no_unique_address]] Second NZNoUnique;
  char B;
};
Foo I;

// CHECK-LABEL:          0 | class SecondEmpty (empty)
// CHECK-NEXT:           0 |   class Empty (base) (empty)
// CHECK-NEXT:             | [sizeof=1, dsize=0, align=1,
// CHECK-NEXT:             |  nvsize=1, nvalign=1]
class SecondEmpty: Empty {
};

// CHECK-LABEL:          0 | class Bar
// CHECK-NEXT:           0 |   class Empty (base) (empty)
// CHECK-NEXT:           1 |   class SecondEmpty ZNoUnique (empty)
// CHECK-NEXT:           1 |     class Empty (base) (empty)
// CHECK-NEXT:           0 |   char C
// CHECK-NEXT:             | [sizeof=2, dsize=1, align=1,
// CHECK-NEXT:             |  nvsize=2, nvalign=1]

// CHECK-LABEL:  LLVMType:%class.Bar = type { i8, i8 }
// CHECK-NEXT:   NonVirtualBaseLLVMType:%class.Bar = type { i8, i8 }
class Bar : Empty {
  [[no_unique_address]] SecondEmpty ZNoUnique;
  char C;
};
Bar J;

// CHECK-LABEL:          0 | class IntFieldClass
// CHECK-NEXT:           0 |   class Empty (base) (empty)
// CHECK-NEXT:           2 |   class Second Field
// CHECK-NEXT:           2 |     class Empty (base) (empty)
// CHECK-NEXT:       2:0-0 |     short A
// CHECK-NEXT:           4 |   int C
// CHECK-NEXT:             | [sizeof=8, dsize=8, align=4,
// CHECK-NEXT:             |  nvsize=8, nvalign=4]

// CHECK-LABEL:   LLVMType:%class.IntFieldClass = type { [2 x i8], %class.Second.base, i32 }
// CHECK-NEXT:    NonVirtualBaseLLVMType:%class.IntFieldClass = type { [2 x i8], %class.Second.base, i32 }
class IntFieldClass : Empty {
  [[no_unique_address]] Second Field;
  int C;
};
IntFieldClass K;

// CHECK-LABEL:         0 | class UnionClass
// CHECK-NEXT:          0 |   class Empty (base) (empty)
// CHECK-NEXT:          0 |   union UnionClass
// CHECK-NEXT:          0 |     int I
// CHECK-NEXT:          0 |     char C
// CHECK-NEXT:          4 |   int C
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4,
// CHECK-NEXT:            |  nvsize=8, nvalign=4]

// CHECK-LABEL:  LLVMType:%class.UnionClass = type { %union.anon, i32 }
// CHECK-NEXT:   NonVirtualBaseLLVMType:%class.UnionClass = type { %union.anon, i32 }
// CHECK-NEXT:   IsZeroInitializable:1
// CHECK-NEXT:   BitFields:[
// CHECK-NEXT: ]>
class UnionClass : Empty {
  [[no_unique_address]] union {
    int I;
    char C;
  } U;
  int C;
};
UnionClass L;

// CHECK-LABEL:         0 | class EnumClass
// CHECK-NEXT:          0 |   class Empty (base) (empty)
// CHECK-NEXT:          0 |   enum E A
// CHECK-NEXT:          4 |   int C
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4,
// CHECK-NEXT:            |  nvsize=8, nvalign=4]

// CHECK-LABEL:  LLVMType:%class.EnumClass = type { i32, i32 }
// CHECK-NEXT:   NonVirtualBaseLLVMType:%class.EnumClass = type { i32, i32 }
// CHECK-NEXT:   IsZeroInitializable:1
// CHECK-NEXT:   BitFields:[
// CHECK-NEXT: ]>
class EnumClass : Empty {
  [[no_unique_address]] enum class E { X, Y, Z } A;
  int C;
};
EnumClass M;

// CHECK-LABEL:         0 | class NoBaseField
// CHECK-NEXT:          0 |   class Empty (base) (empty)
// CHECK-NEXT:          1 |   class Empty A (empty)
// CHECK-NEXT:          0 |   int B
// CHECK-NEXT:            | [sizeof=4, dsize=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK-LABEL:  LLVMType:%class.NoBaseField = type { i32 }
// CHECK-NEXT:   NonVirtualBaseLLVMType:%class.NoBaseField = type { i32 }
// CHECK-NEXT:   IsZeroInitializable:1
// CHECK-NEXT:   BitFields:[
// CHECK-NEXT: ]>
class NoBaseField : Empty {
  [[no_unique_address]] Empty A;
  int B;
};
NoBaseField N;

// CHECK-LABEL:        0 | class FinalEmpty (empty)
// CHECK-NEXT:           | [sizeof=1, dsize=1, align=1,
// CHECK-NEXT:           |  nvsize=1, nvalign=1]

// CHECK-LABEL:        0 | class FinalClass
// CHECK-NEXT:         0 |   class Empty (base) (empty)
// CHECK-NEXT:         0 |   class FinalEmpty A (empty)
// CHECK-NEXT:         0 |   int B
// CHECK-NEXT:           | [sizeof=4, dsize=4, align=4,
// CHECK-NEXT:           |  nvsize=4, nvalign=4]
class FinalEmpty final {};
class FinalClass final : Empty {
  [[no_unique_address]] FinalEmpty A;
  int B;
} O;


// CHECK-LABEL:        0 | union Union2Class::PaddedUnion
// CHECK-NEXT:         0 |   class Empty A (empty)
// CHECK-NEXT:         0 |   char B
// CHECK-NEXT:           | [sizeof=2, dsize=1, align=2,
// CHECK-NEXT:           |  nvsize=1, nvalign=2]

// CHECK-LABEL:        0 | class Union2Class
// CHECK-NEXT:         0 |   class Empty (base) (empty)
// CHECK-NEXT:         2 |   union Union2Class::PaddedUnion U
// CHECK-NEXT:         2 |     class Empty A (empty)
// CHECK-NEXT:         2 |     char B
// CHECK-NEXT:         3 |   char C
// CHECK-NEXT:           | [sizeof=4, dsize=4, align=2,
// CHECK-NEXT:           |  nvsize=4, nvalign=2]
class Union2Class : Empty {
  [[no_unique_address]] union PaddedUnion {
  private:
    Empty A;
    alignas(2) char B;
  } U;
  char C;
} P;

// CHECK-LABEL:          0 | struct NotEmptyWithBitfield
// CHECK-NEXT:           0 |   class Empty (base) (empty)
// CHECK-NEXT:           0 |   char[2] A
// CHECK-NEXT:       2:0-0 |   int B
// CHECK-NEXT:             | [sizeof=4, dsize=3, align=4,
// CHECK-NEXT:             |  nvsize=3, nvalign=4]

// CHECK-LABEL:          0 | union C::
// CHECK-NEXT:           0 |   short C
// CHECK-NEXT:           0 |   struct NotEmptyWithBitfield A
// CHECK-NEXT:           0 |     class Empty (base) (empty)
// CHECK-NEXT:           0 |     char[2] A
// CHECK-NEXT:       2:0-0 |     int B
// CHECK-NEXT:             | [sizeof=4, dsize=3, align=4,
// CHECK-NEXT:             |  nvsize=3, nvalign=4]

// CHECK-LABEL:          0 | struct C
// CHECK-NEXT:           0 |   union C::
// CHECK-NEXT:           0 |     short C
// CHECK-NEXT:           0 |     struct NotEmptyWithBitfield A
// CHECK-NEXT:           0 |       class Empty (base) (empty)
// CHECK-NEXT:           0 |       char[2] A
// CHECK-NEXT:       2:0-0 |       int B
// CHECK-NEXT:             | [sizeof=4, dsize=3, align=4,
// CHECK-NEXT:             |  nvsize=3, nvalign=4]
struct NotEmptyWithBitfield : Empty {
  char A[2];
  int B : 1;
};
struct C {
  [[no_unique_address]] union {
    short C;
    [[no_unique_address]] NotEmptyWithBitfield A;
  } U;
} Q;
