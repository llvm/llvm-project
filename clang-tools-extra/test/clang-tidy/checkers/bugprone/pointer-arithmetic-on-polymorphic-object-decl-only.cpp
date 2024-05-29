// RUN: %check_clang_tidy %s bugprone-pointer-arithmetic-on-polymorphic-object %t

class Base {
public:  
  virtual ~Base() {}
};

class Derived : public Base {};

class AbstractBase {
public:
  virtual void f() = 0;
  virtual ~AbstractBase() {}
};

class AbstractInherited : public AbstractBase {};

class AbstractOverride : public AbstractInherited {
public:
  void f() override {}
};

void operators() {
  Base *b = new Derived[10];

  b += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic class 'Base', undefined behavior if the pointee is a different class [bugprone-pointer-arithmetic-on-polymorphic-object]

  b = b + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: pointer arithmetic on polymorphic class 'Base', undefined behavior if the pointee is a different class [bugprone-pointer-arithmetic-on-polymorphic-object]

  b++;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic class 'Base', undefined behavior if the pointee is a different class [bugprone-pointer-arithmetic-on-polymorphic-object]

  b[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic class 'Base', undefined behavior if the pointee is a different class [bugprone-pointer-arithmetic-on-polymorphic-object]

  delete[] static_cast<Derived*>(b);
}

void subclassWarnings() {
  Base *b = new Base[10];

  // False positive that's impossible to distinguish without
  // path-sensitive analysis, but the code is bug-prone regardless.
  b += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic class 'Base'

  delete[] b;

  // Common false positive is a class that overrides all parent functions.
  Derived *d = new Derived[10];

  d += 1;
  // no-warning

  delete[] d;
}

void abstractWarnings() {
  // Classes with an abstract member funtion are always matched.
  AbstractBase *ab = new AbstractOverride[10];

  ab += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic class 'AbstractBase'

  delete[] static_cast<AbstractOverride*>(ab);

  AbstractInherited *ai = new AbstractOverride[10];

  // FIXME: Match abstract functions that were not overridden.
  ai += 1;
  // no-warning

  delete[] static_cast<AbstractOverride*>(ai);

  // If all abstract member functions are overridden, the class is not matched.
  AbstractOverride *ao = new AbstractOverride[10];

  ao += 1;
  // no-warning

  delete[] ao;
}

template <typename T>
void templateWarning(T *t) {
  // FIXME: Show the location of the template instantiation in diagnostic.
  t += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic class 'Base'
}

void functionArgument(Base *b) {
  b += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic class 'Base'

  templateWarning(b);
}
