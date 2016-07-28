// RUN: clang-rename -offset=134 -new-name=Bar %s -- -frtti | FileCheck %s

class Baz {
  virtual int getValue() const = 0;
};

class Foo : public Baz {                           // CHECK: class Bar : public Baz {
public:
  int getValue() const {
    return 0;
  }
};

int main() {
  Foo foo;                                         // FIXME: Bar foo; <- this one fails
  const Baz &Reference = foo;
  const Baz *Pointer = &foo;

  dynamic_cast<const Foo &>(Reference).getValue(); // CHECK: dynamic_cast<const Bar &>(Reference).getValue();
  dynamic_cast<const Foo *>(Pointer)->getValue();  // CHECK: dynamic_cast<const Bar *>(Pointer)->getValue();
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of Foo when changing
// this file.
