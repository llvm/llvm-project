// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

// C++20 [temp.spec] 13.9/6:
//   The usual access checking rules do not apply to names in a declaration
//   of an explicit instantiation or explicit specialization, with the
//   exception of names appearing in a function body, default argument,
//   base-clause, member-specification, enumerator-list, or static data member
//   or variable template initializer.
//   [Note : In particular, the template arguments and names used in the
//   function declarator(including parameter types, return types and exception
//   specifications) may be private types or objects that would normally not be
//   accessible. — end note]

class A {
  // expected-note@+1 17{{implicitly declared private here}}
  template <typename T> class B {};
  // expected-note@+1 3{{implicitly declared private here}}
  static constexpr int num1 = 42;

protected:
  // expected-note@+1 13{{declared protected here}}
  class C {};
  // expected-note@+1 2{{declared protected here}}
  static constexpr int num2 = 43;
  static int num4;

public:
  template <typename T> class D {};
  static constexpr int num3 = 44;
};
int A::num4 = 44;

class E : public A {

  // Declarations

  // expected-error@+1 {{is a private member of}}
  template <typename T = A::B<int>> void func1();
  template <typename T = A::C> void func2();
  template <typename T = class A::D<int>> void func3();
  // expected-error@+1 {{is a private member of}}
  template <typename T> A::B<int> func4();
  // expected-error@+1 {{is a private member of}}
  template <typename T> A::B<T> func5();
  template <typename T> class A::C func6();
  template <typename T> class A::D<int> func7();
  // expected-error@+1 2{{is a private member of}}
  template <typename T> void func8(class A::B<T>, int x = A::num1);
  template <typename T> void func9(A::C, A::D<T>, int = A::num3);

  // Specializations inside class declaration
  template <> void func1<A::B<char>>() {}
  template <> void func2<class A::D<char>>() {
  } template <> void func3<class A::C>() {
  }
  template <> class A::B<int> func4<A::B<char>>() { return {}; } template <> A::B<A::D<int>> func5<A::D<int>>() {
    return {};
  }
  template <> class A::C func6<A::C>() { return {}; } template <> A::D<int> func7<char>() {
    return {};
  }
  template <> void func8<char>(class A::B<char>, int) {}
  template <> void func9<A::B<char>>(A::C, A::D<A::B<char>>, int) {}

  // FIXME: Instantiations inside class declaration.
  // don't work correctly.
};

// Definitions

template <typename T> void E::func1() {}
template <typename T> void E::func2() {}
template <typename T> void E::func3() {}
// expected-error@+1 {{is a private member of}}
template <typename T> A::B<int> E::func4() { return {}; }
// expected-error@+1 {{is a private member of}}
template <typename T> A::B<T> E::func5() { return {}; }
template <typename T> A::C E::func6() { return {}; }
template <typename T> A::D<int> E::func7() { return {}; }
// expected-error@+1 {{is a private member of}}
template <typename T> void E::func8(A::B<T>, int) {}
template <typename T> void E::func9(A::C, A::D<T>, int) {}

// Specializations

template <> void E::func1<A::B<int>>() {}
template <> void E::func2<class A::C>() {}
template <> void E::func3<class A::D<int>>() {
} template <> class A::B<int> E::func4<A::B<int>>() {
  return {};
} template <> A::B<A::C> E::func5<A::C>() {
  return {};
}
template <> class A::C E::func6<A::D<int>>() { return {}; } template <> A::D<int> E::func7<int>() {
  return {};
}
template <> void E::func8<int>(class A::B<int>, int) {}
template <> void E::func9<A::C>(A::C, A::D<A::C>, int) {}

// Instantiations

template <> void E::func1<A::B<int>>();
template <> void E::func2<class A::C>();
template <> void E::func3<class A::D<int>>();
template <> class A::B<int> E::func4<A::B<int>>();
template <> A::B<A::C> E::func5<A::C>();
template <> class A::C E::func6<A::D<int>>();
template <> A::D<int> E::func7<int>();
template <> void E::func8<int>(class A::B<int>, int);
template <> void E::func9<A::C>(A::C, A::D<A::C>, int);

//----------------------------------------------------------//

// forward declarations

// expected-error@+1 {{is a protected member of}}
template <typename T> class A::C func1();
// expected-error@+1 {{is a private member of}}
template <typename T> A::B<T> func2();
template <typename T> A::D<T> func3();
// expected-error@+1 {{is a private member of}}
template <typename T> class A::B<int> func4();
template <typename T> void func5();
// expected-error@+1 {{is a private member of}}
template <int x = A::num1> void func6();
// expected-error@+1 {{is a protected member of}}
template <int x = A::num2> void func7();
// expected-error@+1 {{is a protected member of}}
template <typename T> void func8(int x = sizeof(A::C));
// expected-error@+1 {{is a private member of}}
template <typename T> void func9(int x = A::num1);
// expected-error@+2 {{is a private member of}}
// expected-error@+1 {{is a protected member of}}
template <typename T> void func10(class A::B<T>, int x = A::num2);
// expected-error@+1 {{is a protected member of}}
template <typename T> void func11(class A::C, A::D<T>, int = A::num3);
template <typename T> void func12();
template <int x> void func13();
template <typename T, int x> void func14();
template <template <typename> typename T> void func15();
// expected-error@+1 {{is a protected member of}}
template <typename T = A::C> void func16();
// expected-error@+1 {{is a private member of}}
template <typename T = A::B<int>> void func17();
// expected-error@+1 {{is a protected member of}}
template <typename T> auto func18() -> A::C;
template <typename T> T func19();

//----------------------------------------------------------//

// definitions

// expected-error@+1 2{{is a protected member of}}
template <typename T> A::C func1() { A::C x; }
// expected-error@+2 {{is a private member of}}
// expected-error@+1 {{is a protected member of}}
template <typename T> A::B<T> func2() { A::D<A::C> x; }
template <typename T> A::D<T> func3() { A::D<int> x; }
// expected-error@+2 2{{is a private member of}}
// expected-error@+1 {{is a protected member of}}
template <typename T> class A::B<int> func4() { A::B<A::C> x; }

template <typename T>
void func5() {
  // expected-error@+2 {{is a private member of}}
  // expected-error@+1 {{is a protected member of}}
  A::B<A::D<A::C>> x;
  // expected-error@+1 {{is a private member of}}
  A::B<int> x2;
}
template <typename T> void func8(int x) {}
template <typename T> void func9(int x) {}
// expected-error@+1 {{is a private member of}}
template <typename T> void func10(A::B<T>, int x) {}
// expected-error@+1 {{is a protected member of}}
template <typename T> void func11(A::C, A::D<T>, int) {}
template <typename T> void func12() {}
template <int x> void func13() {}
template <typename T, int x> void func14() {}
template <template <typename> typename T> void func15() {}
template <typename T> void func16() {}
template <typename T> void func17() {}
// expected-error@+1 {{is a protected member of}}
template <typename T> auto func18() -> A::C {
  // expected-error@+1 {{is a protected member of}}
  return A::C{};
}
template <typename T> T func19() {
  return T{};
}

//----------------------------------------------------------//

// explicit specializations

template <> A::C func1<A::C>();
template <> A::B<A::C> func2<A::C>();
template <> A::D<A::C> func3<A::C>();
template <> class A::B<int> func4<A::C>();
template <> void func5<A::C>();
template <> void func5<A::B<int>>();
template <> void func5<A::D<A::C>>();
template <> void func5<int>();
template <> void func8<A::C>(int x);
template <> void func9<decltype(A::num1)>(int);
template <> void func10<A::D<int>>(A::B<A::D<int>>, int);
template <> void func11<A::C>(A::C, A::D<A::C>, int);
template <> void func12<class A::B<char>>() {
} template <> void func13<A::num1>() {
}
template <> void func14<A::B<int>, A::num2>() {}
template <> void func15<A::D>() {}
template <> void func16<class A::B<char>>() {
} template <> void func17<A::B<class A::C>>() {
}
template <> auto func18<int>() -> class A::C;
template <> A::B<int> func19<class A::B<int>>();

//----------------------------------------------------------//

// explicit instantiations

template void func10<A::C>(A::B<A::C>, decltype(A::num1));
template void func11<A::B<int>>(A::C, A::D<A::B<int>>, decltype(A::num2));
template void func12<A::C>();
template void func13<A::num2>();
template void func13<A::num3>();
template void func14<A::C, A::num1>();
template void func15<A::B>();
template void func17();
template auto func18<char>() -> A::C;
template class A::C func19<A::C>();

//----------------------------------------------------------//

// Other cases

template <int *x> class StealClass {
  friend int stealFunc() { return *x; }
};

template class StealClass<&A::num4>;
int stealFunc();

int stealFunc2() {
  return stealFunc();
}
