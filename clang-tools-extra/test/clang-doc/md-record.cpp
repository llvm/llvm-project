// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// This test requires Linux due to system-dependent USR for the inner class.
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

void H() {
  class I {};
}

union A { int X; int Y; };

enum B { X, Y };

enum class Bc { A, B };

struct C { int i; };

class D {};

class E {
public:
  E() {}
  ~E() {}

protected:
  void ProtectedMethod();
};

void E::ProtectedMethod() {}

class F : virtual private D, public E {};

class X {
  class Y {};
};

// RUN: clang-doc --format=md --doxygen --public --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./F.md | FileCheck %s --check-prefix CHECK-0
// CHECK-0: # class F
// CHECK-0: *Defined at line 36 of test*
// CHECK-0: Inherits from E, D

// RUN: cat %t/docs/./D.md | FileCheck %s --check-prefix CHECK-1
// CHECK-1: # class D
// CHECK-1: *Defined at line 23 of test*

// RUN: cat %t/docs/./GlobalNamespace.md | FileCheck %s --check-prefix CHECK-2
// CHECK-2: # Global Namespace
// CHECK-2: ## Functions
// CHECK-2: ### void H()
// CHECK-2: *Defined at line 11 of test*
// CHECK-2: ## Enums
// CHECK-2: | enum B |
// CHECK-2: --
// CHECK-2: | X |
// CHECK-2: | Y |
// CHECK-2: *Defined at line 17 of test*
// CHECK-2: | enum class Bc |
// CHECK-2: --
// CHECK-2: | A |
// CHECK-2: | B |
// CHECK-2: *Defined at line 19 of test*

// RUN: cat %t/docs/./E.md | FileCheck %s --check-prefix CHECK-3
// CHECK-3: # class E
// CHECK-3: *Defined at line 25 of test*
// CHECK-3: ## Functions
// CHECK-3: ### void E()
// CHECK-3: *Defined at line 27 of test*
// CHECK-3: ### void ~E()
// CHECK-3: *Defined at line 28 of test*
// CHECK-3: ### void ProtectedMethod()
// CHECK-3: *Defined at line 34 of test*

// RUN: cat %t/docs/./C.md | FileCheck %s --check-prefix CHECK-4
// CHECK-4: # struct C
// CHECK-4: *Defined at line 21 of test*
// CHECK-4: ## Members
// CHECK-4: int i

// RUN: cat %t/docs/./X.md | FileCheck %s --check-prefix CHECK-5
// CHECK-5: # class X
// CHECK-5: *Defined at line 38 of test*

// RUN: cat %t/docs/./A.md | FileCheck %s --check-prefix CHECK-6
// CHECK-6: # union A
// CHECK-6: *Defined at line 15 of test*
// CHECK-6: ## Members
// CHECK-6: int X
// CHECK-6: int Y
