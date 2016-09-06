// RUN: %clang_cc1 < %s -triple %itanium_abi_triple -fms-extensions -emit-llvm -x c++ | FileCheck %s

// Test attribute 'optnone' on methods:
//  -- member functions;
//  -- static member functions.

// Verify that all methods of struct A are associated to the same attribute set.
// The attribute set shall contain attributes 'noinline' and 'optnone'.

struct A {
  // Definition of an optnone static method.
  __attribute__((optnone))
  static int static_optnone_method(int a) {
    return a + a;
  }
  // CHECK: @_ZN1A21static_optnone_methodEi({{.*}}) [[OPTNONE:#[0-9]+]]

  // Definition of an optnone normal method.
  __attribute__((optnone))
  int optnone_method(int a) {
    return a + a + a + a;
  }
  // CHECK: @_ZN1A14optnone_methodEi({{.*}}) [[OPTNONE]]

  // Declaration of an optnone method with out-of-line definition
  // that doesn't say optnone.
  __attribute__((optnone))
  int optnone_decl_method(int a);

  // Methods declared without attribute optnone; the definitions will
  // have attribute optnone, and we verify optnone wins.
  __forceinline static int static_forceinline_method(int a);
  __attribute__((always_inline)) int alwaysinline_method(int a);
  __attribute__((noinline)) int noinline_method(int a);
  __attribute__((minsize)) int minsize_method(int a);
};

void foo() {
  A a;
  A::static_optnone_method(4);
  a.optnone_method(14);
  a.optnone_decl_method(12);
  A::static_forceinline_method(5);
  a.alwaysinline_method(5);
  a.noinline_method(6);
  a.minsize_method(7);
}

// No attribute here, should still be on the definition.
int A::optnone_decl_method(int a) {
  return a;
}
// CHECK: @_ZN1A19optnone_decl_methodEi({{.*}}) [[OPTNONE]]

// optnone implies noinline; therefore attribute noinline is added to
// the set of function attributes.
// forceinline is instead translated as 'always_inline'.
// However 'noinline' wins over 'always_inline' and therefore
// the resulting attributes for this method are: noinline + optnone
__attribute__((optnone))
int A::static_forceinline_method(int a) {
  return a + a + a + a;
}
// CHECK: @_ZN1A25static_forceinline_methodEi({{.*}}) [[OPTNONE]]

__attribute__((optnone))
int A::alwaysinline_method(int a) {
  return a + a + a + a;
}
// CHECK: @_ZN1A19alwaysinline_methodEi({{.*}}) [[OPTNONE]]

// 'noinline' + 'noinline and optnone' = 'noinline and optnone'
__attribute__((optnone))
int A::noinline_method(int a) {
  return a + a + a + a;
}
// CHECK: @_ZN1A15noinline_methodEi({{.*}}) [[OPTNONE]]

// 'optnone' wins over 'minsize'
__attribute__((optnone))
int A::minsize_method(int a) {
  return a + a + a + a;
}
// CHECK: @_ZN1A14minsize_methodEi({{.*}}) [[OPTNONE]]


// Test attribute 'optnone' on methods:
//  -- pure virtual functions
//  -- base virtual and derived virtual
//  -- base virtual but not derived virtual
//  -- optnone methods redefined in override

// A method defined in override doesn't inherit the function attributes of the
// superclass method.

struct B {
  virtual int pure_virtual(int a) = 0;
  __attribute__((optnone))
  virtual int pure_virtual_with_optnone(int a) = 0;

  virtual int base(int a) {
    return a + a + a + a;
  }

  __attribute__((optnone))
  virtual int optnone_base(int a) {
    return a + a + a + a;
  }

  __attribute__((optnone))
  virtual int only_base_virtual(int a) {
    return a + a;
  }
};

struct C : public B {
  __attribute__((optnone))
  virtual int pure_virtual(int a) {
    return a + a + a + a;
  }

  virtual int pure_virtual_with_optnone(int a) {
    return a + a + a + a;
  }

  __attribute__((optnone))
  virtual int base(int a) {
    return a + a;
  }

  virtual int optnone_base(int a) {
    return a + a;
  }

  int only_base_virtual(int a) {
    return a + a + a + a;
  }
};

int bar() {
  C c;
  int result;
  result = c.pure_virtual(3);
  result += c.pure_virtual_with_optnone(2);
  result += c.base(5);
  result += c.optnone_base(7);
  result += c.only_base_virtual(9);
  return result;
}

// CHECK: @_ZN1C12pure_virtualEi({{.*}}) {{.*}} [[OPTNONE]]
// CHECK: @_ZN1C25pure_virtual_with_optnoneEi({{.*}}) {{.*}} [[NORMAL:#[0-9]+]]
// CHECK: @_ZN1C4baseEi({{.*}}) {{.*}} [[OPTNONE]]
// CHECK: @_ZN1C12optnone_baseEi({{.*}}) {{.*}} [[NORMAL]]
// CHECK: @_ZN1C17only_base_virtualEi({{.*}}) {{.*}} [[NORMAL]]
// CHECK: @_ZN1B4baseEi({{.*}}) {{.*}} [[NORMAL]]
// CHECK: @_ZN1B12optnone_baseEi({{.*}}) {{.*}} [[OPTNONE]]
// CHECK: @_ZN1B17only_base_virtualEi({{.*}}) {{.*}} [[OPTNONE]]


// CHECK: attributes [[NORMAL]] =
// CHECK-NOT: noinline
// CHECK-NOT: optnone
// CHECK: attributes [[OPTNONE]] = {{.*}} noinline {{.*}} optnone
