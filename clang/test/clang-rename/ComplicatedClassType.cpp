// Forward declaration.
class Foo; /* Test 1 */               // CHECK: class Bar; /* Test 1 */

class Baz {
  virtual int getValue() const = 0;
};

class Foo : public Baz  { /* Test 2 */// CHECK: class Bar : public Baz {
public:
  Foo(int value = 0) : x(value) {}    // CHECK: Bar(int value = 0) : x(value) {}

  Foo &operator++(int) {              // CHECK: Bar &operator++(int) {
    x++;
    return *this;
  }

  bool operator<(Foo const &rhs) {    // CHECK: bool operator<(Bar const &rhs) {
    return this->x < rhs.x;
  }

  int getValue() const {
    return 0;
  }

private:
  int x;
};

int main() {
  Foo *Pointer = 0;                   // CHECK: Bar *Pointer = 0;
  Foo Variable = Foo(10);             // CHECK: Bar Variable = Bar(10);
  for (Foo it; it < Variable; it++) { // CHECK: for (Bar it; it < Variable; it++) {
  }
  const Foo *C = new Foo();           // CHECK: const Bar *C = new Bar();
  const_cast<Foo *>(C)->getValue();   // CHECK: const_cast<Bar *>(C)->getValue();
  Foo foo;                            // CHECK: Bar foo;
  const Baz &BazReference = foo;
  const Baz *BazPointer = &foo;
  dynamic_cast<const Foo &>(BazReference).getValue();     /* Test 3 */ // CHECK: dynamic_cast<const Bar &>(BazReference).getValue();
  dynamic_cast<const Foo *>(BazPointer)->getValue();      /* Test 4 */ // CHECK: dynamic_cast<const Bar *>(BazPointer)->getValue();
  reinterpret_cast<const Foo *>(BazPointer)->getValue();  /* Test 5 */ // CHECK: reinterpret_cast<const Bar *>(BazPointer)->getValue();
  static_cast<const Foo &>(BazReference).getValue();      /* Test 6 */ // CHECK: static_cast<const Bar &>(BazReference).getValue();
  static_cast<const Foo *>(BazPointer)->getValue();       /* Test 7 */ // CHECK: static_cast<const Bar *>(BazPointer)->getValue();
  return 0;
}

// Test 1.
// RUN: clang-rename -offset=30 -new-name=Bar %s -- -frtti | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=155 -new-name=Bar %s -- -frtti | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=1133 -new-name=Bar %s -- -frtti | sed 's,//.*,,' | FileCheck %s
// Test 4.
// RUN: clang-rename -offset=1266 -new-name=Bar %s -- -frtti | sed 's,//.*,,' | FileCheck %s
// Test 5.
// RUN: clang-rename -offset=1402 -new-name=Bar %s -- -frtti | sed 's,//.*,,' | FileCheck %s
// Test 6.
// RUN: clang-rename -offset=1533 -new-name=Bar %s -- -frtti | sed 's,//.*,,' | FileCheck %s
// Test 7.
// RUN: clang-rename -offset=1665 -new-name=Bar %s -- -frtti | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
