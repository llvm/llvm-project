// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -fdump-record-layouts -std=c++17 %s -o %t | FileCheck %s

// CHECK:       *** Dumping AST Record Layout
// CHECK-NEXT:           0 | class Empty (empty)
// CHECK-NEXT:             | [sizeof=1, dsize=1, align=1,
// CHECK-NEXT:             |  nvsize=1, nvalign=1]
// CHECK:       *** Dumping AST Record Layout
// CHECK-NEXT:           0 | class Second
// CHECK-NEXT:           0 |   class Empty (base) (empty)
// CHECK-NEXT:       0:0-0 |   short A
// CHECK-NEXT:             | [sizeof=2, dsize=1, align=2,
// CHECK-NEXT:             |  nvsize=1, nvalign=2]
// CHECK:       *** Dumping AST Record Layout
// CHECK-NEXT:           0 | class Foo
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

// CHECK:       *** Dumping AST Record Layout
// CHECK-NEXT:           0 | class SecondEmpty (empty)
// CHECK-NEXT:           0 |   class Empty (base) (empty)
// CHECK-NEXT:             | [sizeof=1, dsize=0, align=1,
// CHECK-NEXT:             |  nvsize=1, nvalign=1]
class SecondEmpty: Empty {
};

// CHECK:       *** Dumping AST Record Layout
// CHECK-NEXT:           0 | class Bar
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

// CHECK:       *** Dumping AST Record Layout
// CHECK-NEXT:           0 | class IntFieldClass
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
