// Forward declaration.
class Foo; /* Test 1 */               // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10

class Baz {
  virtual int getValue() const = 0;
};

class Foo : public Baz  { /* Test 2 */// CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
public:
  Foo(int value = 0) : x(value) {}    // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6

  Foo &operator++(int) {              // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
    x++;
    return *this;
  }

  bool operator<(Foo const &rhs) {    // CHECK: rename [[@LINE]]:18 -> [[@LINE]]:21
    return this->x < rhs.x;
  }

  int getValue() const {
    return 0;
  }

private:
  int x;
};

int main() {
  Foo *Pointer = 0;                   // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
  Foo Variable = Foo(10);             // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
                                      // CHECK: rename [[@LINE-1]]:18 -> [[@LINE-1]]:21
  for (Foo it; it < Variable; it++) { // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:11
  }
  const Foo *C = new Foo();           // CHECK: rename [[@LINE]]:9 -> [[@LINE]]:12
                                      // CHECK: rename [[@LINE-1]]:22 -> [[@LINE-1]]:25
  const_cast<Foo *>(C)->getValue();   // CHECK: rename [[@LINE]]:14 -> [[@LINE]]:17
  Foo foo;                            // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
  const Baz &BazReference = foo;
  const Baz *BazPointer = &foo;
  dynamic_cast<const Foo &>(BazReference).getValue();     /* Test 3 */ // CHECK: rename [[@LINE]]:22 -> [[@LINE]]:25
  dynamic_cast<const Foo *>(BazPointer)->getValue();      /* Test 4 */ // CHECK: rename [[@LINE]]:22 -> [[@LINE]]:25
  reinterpret_cast<const Foo *>(BazPointer)->getValue();  /* Test 5 */ // CHECK: rename [[@LINE]]:26 -> [[@LINE]]:29
  static_cast<const Foo &>(BazReference).getValue();      /* Test 6 */ // CHECK: rename [[@LINE]]:21 -> [[@LINE]]:24
  static_cast<const Foo *>(BazPointer)->getValue();       /* Test 7 */ // CHECK: rename [[@LINE]]:21 -> [[@LINE]]:24
  return 0;
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:2:7 -new-name=Bar %s -frtti | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:8:7 -new-name=Bar %s -frtti | FileCheck %s
// Test 3.
// RUN: clang-refactor-test rename-initiate -at=%s:41:22 -new-name=Bar %s -frtti | FileCheck %s
// Test 4.
// RUN: clang-refactor-test rename-initiate -at=%s:42:22 -new-name=Bar %s -frtti | FileCheck %s
// Test 5.
// RUN: clang-refactor-test rename-initiate -at=%s:43:26 -new-name=Bar %s -frtti | FileCheck %s
// Test 6.
// RUN: clang-refactor-test rename-initiate -at=%s:44:21 -new-name=Bar %s -frtti | FileCheck %s
// Test 7.
// RUN: clang-refactor-test rename-initiate -at=%s:45:21 -new-name=Bar %s -frtti | FileCheck %s
