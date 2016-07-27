// RUN: clang-rename -offset=91 -new-name=Bar %s -- | FileCheck %s

class Baz {
};

class Foo : public Baz {                          // CHECK: class Bar : public Baz {
public:
  int getValue() const {
    return 0;
  }
};

int main() {
  Foo foo;                                        // FIXME: Bar foo;
  const Baz &Reference = foo;
  const Baz *Pointer = &foo;

  static_cast<const Foo &>(Reference).getValue(); // CHECK: static_cast<const Bar &>(Reference).getValue();
  static_cast<const Foo *>(Pointer)->getValue();  // CHECK: static_cast<const Bar *>(Pointer)->getValue();
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of foo when changing
// this file.
