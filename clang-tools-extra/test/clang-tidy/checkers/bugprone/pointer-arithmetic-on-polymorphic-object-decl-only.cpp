// RUN: %check_clang_tidy %s bugprone-pointer-arithmetic-on-polymorphic-object %t -- \
// RUN: -config="{CheckOptions: \
// RUN: {bugprone-pointer-arithmetic-on-polymorphic-object.IgnoreInheritedVirtualFunctions: true}}"

class Base {
public:  
  virtual ~Base() {}
};

class Derived : public Base {};

class FinalDerived final : public Base {};

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
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic object of type 'Base' can result in undefined behavior if the dynamic type differs from the pointer type [bugprone-pointer-arithmetic-on-polymorphic-object]

  b = b + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: pointer arithmetic on polymorphic object of type 'Base' can result in undefined behavior if the dynamic type differs from the pointer type [bugprone-pointer-arithmetic-on-polymorphic-object]

  b++;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic object of type 'Base' can result in undefined behavior if the dynamic type differs from the pointer type [bugprone-pointer-arithmetic-on-polymorphic-object]

  --b;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: pointer arithmetic on polymorphic object of type 'Base' can result in undefined behavior if the dynamic type differs from the pointer type [bugprone-pointer-arithmetic-on-polymorphic-object]

  b[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic object of type 'Base' can result in undefined behavior if the dynamic type differs from the pointer type [bugprone-pointer-arithmetic-on-polymorphic-object]

  delete[] static_cast<Derived*>(b);
}

void subclassWarnings() {
  Base *b = new Base[10];

  // False positive that's impossible to distinguish without
  // path-sensitive analysis, but the code is bug-prone regardless.
  b += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic object of type 'Base'

  delete[] b;

  // Common false positive is a class that overrides all parent functions.
  Derived *d = new Derived[10];

  d += 1;
  // no-warning

  delete[] d;

  // Final classes cannot have a dynamic type.
  FinalDerived *fd = new FinalDerived[10];

  fd += 1;
  // no-warning

  delete[] fd;
}

void abstractWarnings() {
  // Classes with an abstract member funtion are always matched.
  AbstractBase *ab = new AbstractOverride[10];

  ab += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic object of type 'AbstractBase'

  delete[] static_cast<AbstractOverride*>(ab);

  AbstractInherited *ai = new AbstractOverride[10];

  ai += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic object of type 'AbstractInherited'

  delete[] static_cast<AbstractOverride*>(ai);

  // If all abstract member functions are overridden, the class is not matched.
  AbstractOverride *ao = new AbstractOverride[10];

  ao += 1;
  // no-warning

  delete[] ao;
}

template <typename T>
void templateWarning(T *t) {
  // FIXME: Tidy doesn't support template instantiation locations properly.
  t += 1;
  // no-warning
}

void functionArgument(Base *b) {
  b += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic object of type 'Base'

  templateWarning(b);
}

using BaseAlias = Base;
using DerivedAlias = Derived;
using FinalDerivedAlias = FinalDerived;

using BasePtr = Base*;
using DerivedPtr = Derived*;
using FinalDerivedPtr = FinalDerived*;

void typeAliases(BaseAlias *b, DerivedAlias *d, FinalDerivedAlias *fd,
                 BasePtr bp, DerivedPtr dp, FinalDerivedPtr fdp) {
  b += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic object of type 'Base'

  d += 1;
  // no-warning

  fd += 1;
  // no-warning

  bp += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on polymorphic object of type 'Base'

  dp += 1;
  // no-warning

  fdp += 1;
  // no-warning
}
