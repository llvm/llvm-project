// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++20 [temp.class.spec] 13.7.5/10
//   The usual access checking rules do not apply to non-dependent names
//   used to specify template arguments of the simple-template-id of the
//   partial specialization.
//
// C++20 [temp.spec] 13.9/6:
//   The usual access checking rules do not apply to names in a declaration
//   of an explicit instantiation or explicit specialization...

// TODO: add test cases for `enum`

// class for tests
class TestClass {
public:
  class PublicClass {};
  template <class T> class TemplatePublicClass {};

  using AliasPublicClass = unsigned char;

  void publicFunc();
  void publicFuncOverloaded();
  void publicFuncOverloaded(int);

  static void publicStaticFunc();
  static void publicStaticFuncOverloaded();
  static void publicStaticFuncOverloaded(int);

  static constexpr int publicStaticInt = 42;

protected:
  // expected-note@+1 8{{declared protected here}}
  class ProtectedClass {};
  template <class T> class TemplateProtectedClass {};

  // expected-note@+1 2{{declared protected here}}
  using AliasProtectedClass = const char;

  // expected-note@+1 3{{declared protected here}}
  void protectedFunc();
  void protectedFuncOverloaded();
  void protectedFuncOverloaded(int);

  // expected-note@+1 2{{declared protected here}}
  static void protectedStaticFunc();
  // expected-note@+1 2{{declared protected here}}
  static void protectedStaticFuncOverloaded();
  static void protectedStaticFuncOverloaded(int);

  // expected-note@+1 2{{declared protected here}}
  static constexpr int protectedStaticInt = 43;

private:
  // expected-note@+1 10{{declared private here}}
  class PrivateClass {};
  // expected-note@+1 {{declared private here}}
  template <class T> class TemplatePrivateClass {};

  using AliasPrivateClass = char *;

  void privateFunc();
  void privateFuncOverloaded();
  void privateFuncOverloaded(int);

  static void privateStaticFunc();
  static void privateStaticFuncOverloaded();
  static void privateStaticFuncOverloaded(int);

  static constexpr int privateStaticInt = 44;
};

void globalFunction() {}

//----------------------------------------------------------//

// template declarations for explicit instantiations
template <typename T> class IT1 {};
template <typename T1, typename T2> class IT2 {};
template <int X> class IT3 {};
template <void (TestClass::*)()> class IT4 {};
template <void (*)()> class IT5 {};
template <typename T> class IT6 {
  template <typename NT> class NIT1 {};
};
template <typename T1, typename T2> class IT7 {};
template <void (TestClass::*)(), int X> class IT8 {};
template <typename T, void (*)()> class IT9 {};

// explicit instantiations

// public
template class IT1<TestClass::PublicClass>;
template struct IT1<TestClass::TemplatePublicClass<int>>;
template class IT1<TestClass::AliasPublicClass>;
template struct IT2<TestClass::PublicClass, TestClass::PublicClass>;
template class IT3<TestClass::publicStaticInt>;
template struct IT4<&TestClass::publicFunc>;
template class IT4<&TestClass::publicFuncOverloaded>;
template class IT5<&TestClass::publicStaticFunc>;
template class IT5<&TestClass::publicStaticFuncOverloaded>;
template class IT5<&globalFunction>;
template class IT6<TestClass::PublicClass>::template NIT1<TestClass::PublicClass>;
template class IT7<TestClass::AliasPublicClass, TestClass::PublicClass>;
template struct IT7<TestClass::PublicClass, TestClass::TemplatePublicClass<TestClass::PublicClass>>;
template class IT8<&TestClass::publicFunc, TestClass::publicStaticInt>;
template class IT8<&TestClass::publicFuncOverloaded, TestClass::publicStaticInt>;
template class IT9<TestClass::PublicClass, &TestClass::publicStaticFunc>;
template class IT9<TestClass::PublicClass, &TestClass::publicStaticFuncOverloaded>;
template class IT9<TestClass::PublicClass, &globalFunction>;

// protected
template class IT1<TestClass::ProtectedClass>;
template struct IT1<TestClass::TemplateProtectedClass<int>>;
template class IT1<TestClass::AliasProtectedClass>;
template struct IT2<TestClass::ProtectedClass, TestClass::ProtectedClass>;
template class IT3<TestClass::protectedStaticInt>;
template struct IT4<&TestClass::protectedFunc>;
template class IT4<&TestClass::protectedFuncOverloaded>;
template class IT5<&TestClass::protectedStaticFunc>;
template class IT5<&TestClass::protectedStaticFuncOverloaded>;
template class IT6<TestClass::ProtectedClass>::template NIT1<TestClass::ProtectedClass>;
template class IT7<TestClass::AliasProtectedClass, TestClass::ProtectedClass>;
template struct IT7<TestClass::ProtectedClass, TestClass::TemplateProtectedClass<TestClass::ProtectedClass>>;
template class IT8<&TestClass::protectedFunc, TestClass::protectedStaticInt>;
template class IT8<&TestClass::protectedFuncOverloaded, TestClass::protectedStaticInt>;
template class IT9<TestClass::ProtectedClass, &TestClass::protectedStaticFunc>;
template class IT9<TestClass::ProtectedClass, &TestClass::protectedStaticFuncOverloaded>;
template class IT9<TestClass::ProtectedClass, &globalFunction>;

// private
template class IT1<TestClass::PrivateClass>;
template struct IT1<TestClass::TemplatePrivateClass<int>>;
template class IT1<TestClass::AliasPrivateClass>;
template struct IT2<TestClass::PrivateClass, TestClass::PrivateClass>;
template class IT3<TestClass::privateStaticInt>;
template struct IT4<&TestClass::privateFunc>;
template class IT4<&TestClass::privateFuncOverloaded>;
template class IT5<&TestClass::privateStaticFunc>;
template class IT5<&TestClass::privateStaticFuncOverloaded>;
template class IT6<TestClass::PrivateClass>::template NIT1<TestClass::PrivateClass>;
template class IT7<TestClass::AliasPrivateClass, TestClass::PrivateClass>;
template struct IT7<TestClass::PrivateClass, TestClass::TemplatePrivateClass<TestClass::PrivateClass>>;
template class IT8<&TestClass::privateFunc, TestClass::privateStaticInt>;
template class IT8<&TestClass::privateFuncOverloaded, TestClass::privateStaticInt>;
template class IT9<TestClass::PrivateClass, &TestClass::privateStaticFunc>;
template class IT9<TestClass::PrivateClass, &TestClass::privateStaticFuncOverloaded>;
template class IT9<TestClass::PrivateClass, &globalFunction>;

//----------------------------------------------------------//

// template declarations for full specializations
template <typename T> class CT1 {};
template <typename T1, typename T2> class CT2 {};
template <int X> class CT3 {};
template <void (TestClass::*)()> class CT4 {};
template <void (*)()> class CT5 {};
template <typename T> class CT6 {
  template <typename NT> class NCT1 {};
  template <typename NT> class NCT2; // forward declaration
};

// full specializations

// public
template <> class CT1<TestClass::PublicClass>;
template <typename T> class CT1<TestClass::TemplatePublicClass<T>>; // not full but let it be here
template <> struct CT1<TestClass::TemplatePublicClass<int>>;
template <> class CT1<TestClass::AliasPublicClass>;
template <> struct CT2<TestClass::PublicClass, TestClass::PublicClass>;
template <> class CT3<TestClass::publicStaticInt>;
template <> struct CT4<&TestClass::publicFunc>;
template <> class CT4<&TestClass::publicFuncOverloaded>;
template <> struct CT5<&TestClass::publicStaticFunc>;
template <> class CT5<&TestClass::publicStaticFuncOverloaded>;
template <> class CT5<&globalFunction>;
template <> template <> class CT6<TestClass::PublicClass>::NCT1<TestClass::PublicClass>;

template <> class CT1<TestClass::PublicClass> final {};
template <typename T> class CT1<TestClass::TemplatePublicClass<T>> {};
template <> class CT1<TestClass::TemplatePublicClass<int>> final {};
template <> class CT1<TestClass::AliasPublicClass> {};
template <> class CT2<TestClass::PublicClass, TestClass::PublicClass> final {};
template <> class CT3<TestClass::publicStaticInt> {};
template <> class CT4<&TestClass::publicFunc> final {};
template <> class CT4<&TestClass::publicFuncOverloaded> {};
template <> class CT5<&TestClass::publicStaticFunc> final {};
template <> class CT5<&TestClass::publicStaticFuncOverloaded> {};
template <> class CT5<&globalFunction> final {};
template <> template <> class CT6<TestClass::PublicClass>::NCT1<TestClass::PublicClass> {};
template <> template <typename NT> class CT6<TestClass::PublicClass>::NCT2 final {}; // declaration

// protected
template <> class CT1<TestClass::ProtectedClass>;
template <typename T> class CT1<TestClass::TemplateProtectedClass<T>>; // not full but let it be here
template <> class CT1<TestClass::TemplateProtectedClass<int>>;
template <> struct CT1<TestClass::AliasProtectedClass>;
template <> class CT2<TestClass::ProtectedClass, TestClass::ProtectedClass>;
template <> struct CT3<TestClass::protectedStaticInt>;
template <> class CT4<&TestClass::protectedFunc>;
template <> struct CT4<&TestClass::protectedFuncOverloaded>;
template <> class CT5<&TestClass::protectedStaticFunc>;
template <> class CT5<&TestClass::protectedStaticFuncOverloaded>;
template <> template <> class CT6<TestClass::ProtectedClass>::NCT1<TestClass::ProtectedClass>;

template <> class CT1<TestClass::ProtectedClass> {};
template <typename T> class CT1<TestClass::TemplateProtectedClass<T>> final {}; // not full but let it be here
template <> class CT1<TestClass::TemplateProtectedClass<int>> {};
template <> class CT1<TestClass::AliasProtectedClass> final {};
template <> class CT2<TestClass::ProtectedClass, TestClass::ProtectedClass> {};
template <> class CT3<TestClass::protectedStaticInt> final {};
template <> class CT4<&TestClass::protectedFunc> {};
template <> class CT4<&TestClass::protectedFuncOverloaded> final {};
template <> class CT5<&TestClass::protectedStaticFunc> {};
template <> class CT5<&TestClass::protectedStaticFuncOverloaded> final {};
template <> template <> class CT6<TestClass::ProtectedClass>::NCT1<TestClass::ProtectedClass> {};
template <> template <typename NT> class CT6<TestClass::ProtectedClass>::NCT2 final {}; // declaration

// private
template <> class CT1<TestClass::PrivateClass>;
template <typename T> class CT1<TestClass::TemplatePrivateClass<T>>; // not full but let it be here
template <> struct CT1<TestClass::TemplatePrivateClass<int>>;
template <> class CT1<TestClass::AliasPrivateClass>;
template <> struct CT2<TestClass::PrivateClass, TestClass::PrivateClass>;
template <> class CT3<TestClass::privateStaticInt>;
template <> struct CT4<&TestClass::privateFunc>;
template <> class CT4<&TestClass::privateFuncOverloaded>;
template <> class CT5<&TestClass::privateStaticFunc>;
template <> class CT5<&TestClass::privateStaticFuncOverloaded>;
template <> template <> class CT6<TestClass::PrivateClass>::NCT1<TestClass::PrivateClass>;

template <> class CT1<TestClass::PrivateClass> final {};
template <typename T> class CT1<TestClass::TemplatePrivateClass<T>> {}; // not full but let it be here
template <> class CT1<TestClass::TemplatePrivateClass<int>> final {};
template <> class CT1<TestClass::AliasPrivateClass> {};
template <> class CT2<TestClass::PrivateClass, TestClass::PrivateClass> final {};
template <> class CT3<TestClass::privateStaticInt> {};
template <> class CT4<&TestClass::privateFunc> final {};     // PR37424
template <> class CT4<&TestClass::privateFuncOverloaded> {}; // PR37424
template <> class CT5<&TestClass::privateStaticFunc> final {};
template <> class CT5<&TestClass::privateStaticFuncOverloaded> {};
template <> template <> class CT6<TestClass::PrivateClass>::NCT1<TestClass::PrivateClass> final {};
template <> template <typename NT> class CT6<TestClass::PrivateClass>::NCT2 {}; // declaration

//----------------------------------------------------------//

// template declarations for full specializations with parents
class P1 {};
template <typename T> class PCT1 {};
template <typename T1, typename T2> class PCT2 {};
template <int X> class PCT3 {};
template <void (TestClass::*)()> class PCT4 {};
template <void (*)()> class PCT5 {}; // expected-note {{template parameter is declared here}}
template <typename T> class PCT6 {
  // expected-note@+1 3{{implicitly declared private here}}
  template <typename NT> class NPCT1 {};
  // expected-note@+1 {{template is declared here}}
  template <typename NT> class NPCT2; // forward declaration
};

// full specializations with parents

// protected + public
template <> class PCT1<TestClass::PublicClass> : P1 {};
template <typename T> class PCT1<TestClass::TemplatePublicClass<T>> : PCT2<TestClass::PublicClass, TestClass::PublicClass> {}; // not full but let it be here
template <> struct PCT1<TestClass::TemplatePublicClass<int>> : PCT1<TestClass::AliasPublicClass> {};
template <> class PCT1<TestClass::AliasProtectedClass> : PCT2<TestClass::PublicClass, int> {};
template <> struct PCT2<TestClass::ProtectedClass, TestClass::PublicClass> : PCT3<TestClass::publicStaticInt> {};
template <> class PCT3<TestClass::protectedStaticInt> : PCT4<&TestClass::publicFunc> {};
template <> struct PCT4<&TestClass::protectedFunc> : PCT5<&TestClass::publicStaticFunc> {};
template <> class PCT4<&TestClass::publicFuncOverloaded> : PCT5<&TestClass::publicStaticFuncOverloaded> {};
template <> class PCT5<&TestClass::protectedStaticFunc> : PCT5<&TestClass::publicStaticFuncOverloaded> {};
// expected-error@+1 {{is a private member of}}
template <> class PCT5<&TestClass::protectedStaticFuncOverloaded> : PCT6<TestClass::PublicClass>::NPCT1<TestClass::PublicClass> {};
// expected-error@+2 {{is a protected member of}}
// expected-error@+1 {{is a private member of}}
template <> class PCT5<&globalFunction> : PCT6<TestClass::ProtectedClass>::NPCT1<int> {};
template <> template <typename NT> class PCT6<TestClass::PublicClass>::NPCT2 : P1 {}; // declaration
template <> template <> class PCT6<TestClass::PublicClass>::NPCT1<TestClass::ProtectedClass> : PCT6<TestClass::PublicClass>::template NPCT2<int> {};

// protected + private
template <> class PCT1<TestClass::PrivateClass> : P1 {};
// expected-error@+2 {{is a protected member of}}
// expected-error@+1 {{is a private member of}}
template <typename T> class PCT1<TestClass::TemplatePrivateClass<T>> : PCT2<TestClass::PrivateClass, TestClass::ProtectedClass> {}; // not full but let it be here
// expected-error@+1 {{is a protected member of}}
template <> class PCT1<TestClass::TemplatePrivateClass<int>> : PCT1<TestClass::AliasProtectedClass> {};
// expected-error@+2 {{is a protected member of}}
// expected-error@+1 {{is a private member of}}
template <> class PCT1<TestClass::AliasPrivateClass> : PCT2<TestClass::ProtectedClass, TestClass::PrivateClass> {};
// expected-error@+1 {{is a protected member of}}
template <> class PCT2<TestClass::PrivateClass, TestClass::PrivateClass> : PCT3<TestClass::protectedStaticInt> {};
// expected-error@+1 {{is a protected member of}}
template <> class PCT3<TestClass::privateStaticInt> : PCT4<&TestClass::protectedFunc> {};
// expected-error@+1 {{is a protected member of}}
template <> class PCT4<&TestClass::privateFunc> : PCT5<&TestClass::protectedStaticFunc> {};
// expected-error@+1 {{is a protected member of}}
template <> class PCT4<&TestClass::privateFuncOverloaded> : PCT5<&TestClass::protectedStaticFuncOverloaded> {};
template <> class PCT5<&TestClass::privateStaticFunc> : P1 {};
// expected-error@+2 {{implicit instantiation of undefined template}}
// expected-error@+1 {{is a private member of}}
template <> template <> class PCT6<TestClass::PrivateClass>::NPCT1<TestClass::PrivateClass> : PCT6<TestClass::PrivateClass>::NPCT2<int> {};
// expected-error@+1 3{{is a private member of}}
template <> class PCT5<&TestClass::privateStaticFuncOverloaded> : PCT6<TestClass::PrivateClass>::NPCT1<TestClass::PrivateClass> {};
template <> template <typename NT> class PCT6<TestClass::PrivateClass>::NPCT2 : P1 {}; // declaration

//----------------------------------------------------------//

// template declarations for partial specializations
template <typename T1, typename T2> class CTT1 {};
template <typename T1, typename T2, typename T3> class CTT2 {};
template <typename T, int X> class CTT3 {};
template <typename T, void (TestClass::*)()> class CTT4 {};
template <typename T, void (*)()> class CTT5 {};
template <typename T1, typename T2> class CTT6 {
  template <typename NT> class NCT1 {};
  template <typename NT> class NCT2; // forward declaration
  template <typename NT1, typename NT2> class NCT3 {};
  template <typename NT1, typename NT2> class NCT4; // forward declaration
};

// partial specializations

// public
template <typename T> class CTT1<T, TestClass::PublicClass> final {};
template <typename T> class CTT1<T, TestClass::TemplatePublicClass<T>> {};
template <typename T> struct CTT1<T, TestClass::TemplatePublicClass<int>> final {};
template <typename T> class CTT1<T, TestClass::AliasPublicClass> {};
template <typename T> struct CTT2<T, TestClass::PublicClass, TestClass::PublicClass> final {};
template <typename T> struct CTT2<TestClass::PublicClass, T, TestClass::PublicClass> {};
template <typename T> class CTT2<TestClass::PublicClass, TestClass::PublicClass, T> final {};
template <typename T> class CTT3<T, TestClass::publicStaticInt> {};
template <typename T> class CTT4<T, &TestClass::publicFunc> final {};
template <typename T> class CTT4<T, &TestClass::publicFuncOverloaded> {};
template <typename T> class CTT5<T, &TestClass::publicStaticFunc> final {};
template <typename T> class CTT5<T, &TestClass::publicStaticFuncOverloaded> {};
template <typename T> class CTT5<T, &globalFunction> final {};
// expected-error@+1 {{cannot specialize a dependent template}}
template <typename T1> template <typename T2> class CTT6<T1, TestClass::PublicClass>::template NCT1<T2 *> {};
template <typename T1, typename T2> template <typename T3> class CTT6<T1, T2>::NCT1<T3 *> final {};
template <typename T1, typename T2> template <typename T3> class CTT6<T1, T2>::NCT2 {}; // declaration
template <typename T1, typename T2> template <typename T3> class CTT6<T1, T2>::NCT2<T3 *> final {};
template <typename T1, typename T2> template <typename T3> class CTT6<T1, T2>::NCT3<T3, TestClass::PublicClass> {};
// expected-error@+1 {{cannot specialize a dependent template}}
template <typename T1> template <typename T2> class CTT6<T1, TestClass::PublicClass>::template NCT3<T2, TestClass::PublicClass> final {};
template <typename T1, typename T2> template <typename T3, typename T4> class CTT6<T1, T2>::NCT4 {}; // declaration
template <typename T1, typename T2> template <typename T3> class CTT6<T1, T2>::NCT4<T3, TestClass::PublicClass> final {};
template <typename T> class CTT6<TestClass::PublicClass, T> {
  template <typename T1, typename T2> class NCT3 {};
  template <typename T1, typename T2> class NCT4;
};
template <typename T1> template <typename T2> class CTT6<TestClass::PublicClass, T1>::NCT3<T2, TestClass::PublicClass> {};
template <typename T1> template <typename, typename> class CTT6<TestClass::PublicClass, T1>::NCT4 final {};
template <typename T1> template <typename T2> class CTT6<TestClass::PublicClass, T1>::NCT4<T2, TestClass::PublicClass> {};

// protected

template <typename T> class CTT1<T, TestClass::ProtectedClass> {};
template <typename T> class CTT1<T, TestClass::TemplateProtectedClass<T>> final {};
template <typename T> struct CTT1<T, TestClass::TemplateProtectedClass<int>> {};
template <typename T> class CTT1<T, TestClass::AliasProtectedClass> final {};
template <typename T> struct CTT2<T, TestClass::ProtectedClass, TestClass::ProtectedClass> {};
template <typename T> class CTT2<TestClass::ProtectedClass, T, TestClass::ProtectedClass> final {};
template <typename T> struct CTT2<TestClass::ProtectedClass, TestClass::ProtectedClass, T> {};
template <typename T> class CTT3<T, TestClass::protectedStaticInt> final {};
template <typename T> class CTT4<T, &TestClass::protectedFunc> {};
template <typename T> class CTT4<T, &TestClass::protectedFuncOverloaded> final {};
template <typename T> class CTT5<T, &TestClass::protectedStaticFunc> {};
template <typename T> class CTT5<T, &TestClass::protectedStaticFuncOverloaded> final {};
// expected-error@+1 {{cannot specialize a dependent template}}
template <typename T1> template <typename T2> class CTT6<T1, TestClass::ProtectedClass>::template NCT1<T2 *> {};
template <typename T1, typename T2> template <typename T3> class CTT6<T1, T2>::NCT3<T3, TestClass::ProtectedClass> final {};
// expected-error@+1 {{cannot specialize a dependent template}}
template <typename T1> template <typename T2> class CTT6<T1, TestClass::ProtectedClass>::template NCT3<T2, TestClass::ProtectedClass> {};
template <typename T1, typename T2> template <typename T3> class CTT6<T1, T2>::NCT4<T3, TestClass::ProtectedClass> final {};
template <typename T> class CTT6<TestClass::ProtectedClass, T> {
  template <typename T1, typename T2> class NCT3 {};
  template <typename T1, typename T2> class NCT4;
};
template <typename T1> template <typename T2> class CTT6<TestClass::ProtectedClass, T1>::NCT3<T2, TestClass::ProtectedClass> final {};
template <typename T1> template <typename, typename> class CTT6<TestClass::ProtectedClass, T1>::NCT4 {};
template <typename T1> template <typename T2> class CTT6<TestClass::ProtectedClass, T1>::NCT4<T2, TestClass::ProtectedClass> final {};

// private

template <typename T> class CTT1<T, TestClass::PrivateClass> final {};
template <typename T> class CTT1<T, TestClass::TemplatePrivateClass<T>> {};
template <typename T> struct CTT1<T, TestClass::TemplatePrivateClass<int>> final {};
template <typename T> class CTT1<T, TestClass::AliasPrivateClass> {};
template <typename T> struct CTT2<T, TestClass::PrivateClass, TestClass::PrivateClass> final {};
template <typename T> class CTT2<TestClass::PrivateClass, T, TestClass::PrivateClass> {};
template <typename T> struct CTT2<TestClass::PrivateClass, TestClass::PrivateClass, T> final {};
template <typename T> class CTT3<T, TestClass::privateStaticInt> {};
template <typename T> class CTT4<T, &TestClass::privateFunc> final {};
template <typename T> class CTT4<T, &TestClass::privateFuncOverloaded> {};
template <typename T> class CTT5<T, &TestClass::privateStaticFunc> final {};
template <typename T> class CTT5<T, &TestClass::privateStaticFuncOverloaded> {};
// expected-error@+1 {{cannot specialize a dependent template}}
template <typename T1> template <typename T2> class CTT6<T1, TestClass::PrivateClass>::template NCT1<T2 *> final {};
template <typename T1, typename T2> template <typename T3> class CTT6<T1, T2>::NCT3<T3, TestClass::PrivateClass> {};
// expected-error@+1 {{cannot specialize a dependent template}}
template <typename T1> template <typename T2> class CTT6<T1, TestClass::PrivateClass>::template NCT3<T2, TestClass::PrivateClass> final {};
template <typename T1, typename T2> template <typename T3> class CTT6<T1, T2>::NCT4<T3, TestClass::PrivateClass> {};
template <typename T> class CTT6<TestClass::PrivateClass, T> {
  template <typename T1, typename T2> class NCT3 {};
  template <typename T1, typename T2> class NCT4;
};
template <typename T1> template <typename T2> class CTT6<TestClass::PrivateClass, T1>::NCT3<T2, TestClass::PrivateClass> {};
template <typename T1> template <typename, typename> class CTT6<TestClass::PrivateClass, T1>::NCT4 final {};
template <typename T1> template <typename T2> class CTT6<TestClass::PrivateClass, T1>::NCT4<T2, TestClass::PrivateClass> final {};

//----------------------------------------------------------//

// template declarations for partial specializations with parents
template <typename T1, typename T2> class PCTT1 {};
template <typename T1, typename T2, typename T3> class PCTT2 {};
template <typename T, int X> class PCTT3 {};
template <typename T, void (TestClass::*)()> class PCTT4 {};
template <typename T, void (*)()> class PCTT5 {}; // expected-note {{template parameter is declared here}}
template <typename T1, typename T2> class PCTT6 {
  template <typename NT> class NCT1 {};
  template <typename NT> class NCT2; // forward declaration
  template <typename NT1, typename NT2> class NCT3 {};
  template <typename NT1, typename NT2> class NCT4; // forward declaration
};

// partial specializations with parents

// protected + public
template <typename T> class PCTT1<T, TestClass::PublicClass> : P1 {};
template <typename T> struct PCTT1<T, TestClass::TemplatePublicClass<T>> final : PCTT2<T, TestClass::PublicClass, TestClass::PublicClass> {}; // not full but let it be here
template <typename T> class PCTT1<T, TestClass::TemplatePublicClass<int>> : PCTT1<T, TestClass::AliasPublicClass> {};
// expected-error@+1 {{is a protected member of}}
template <typename T> class PCTT1<T, TestClass::TemplatePublicClass<TestClass::TemplateProtectedClass<T>>> final : PCTT1<T, TestClass::ProtectedClass> {};
template <typename T> struct PCTT1<T, TestClass::AliasProtectedClass> : PCTT2<T, TestClass::PublicClass, int> {};
template <typename T> class PCTT2<T, TestClass::ProtectedClass, TestClass::PublicClass> final : PCTT3<T, TestClass::publicStaticInt> {};
template <typename T> class PCTT3<T, TestClass::protectedStaticInt> : PCTT4<T, &TestClass::publicFunc> {};
template <typename T> struct PCTT4<T, &TestClass::protectedFunc> final : PCTT5<T, &TestClass::publicStaticFunc> {};
template <typename T> class PCTT4<T, &TestClass::publicFuncOverloaded> : PCTT5<T, &TestClass::publicStaticFuncOverloaded> {};
template <typename T> class PCTT5<T, &TestClass::protectedStaticFunc> final : PCTT5<T, &TestClass::publicStaticFuncOverloaded> {};
template <typename T> class PCTT5<T, &TestClass::protectedStaticFuncOverloaded> : PCTT6<T, TestClass::PublicClass>::template NCT1<TestClass::PublicClass> {};
// expected-error@+1 {{is a protected member of}}
template <typename T> class PCTT5<T, &globalFunction> : PCTT6<T, TestClass::ProtectedClass>::template NCT1<int> {};
// expected-error@+1 {{is a protected member of}}
template <typename T1, typename T2> template <typename T3> class PCTT6<T1, T2>::NCT2 final : PCTT4<T1, &TestClass::protectedFunc> {}; // declaration
template <typename T1, typename T2> template <typename T3> class PCTT6<T1, T2>::NCT2<T3 *> : P1 {};
// expected-error@+2 {{cannot specialize a dependent template}}
// expected-error@+1 {{is a protected member of}}
template <typename T> template <typename NT> class PCTT6<T, TestClass::ProtectedClass>::template NCT1<NT *> : PCTT6<T, TestClass::ProtectedClass>::template NCT2<int> {};

// protected + private
template <typename T> class PCTT1<T, TestClass::PrivateClass> : P1 {};
// expected-error@+2 {{is a protected member of}}
// expected-error@+1 {{is a private member of}}
template <typename T> struct PCTT1<T, TestClass::TemplatePrivateClass<T>> final : PCTT2<T, TestClass::PrivateClass, TestClass::ProtectedClass> {}; // not full but let it be here
// expected-error@+1 {{is a protected member of}}
template <typename T> class PCTT1<T, TestClass::TemplatePrivateClass<int>> : PCTT1<T, TestClass::AliasProtectedClass> {};
// expected-error@+2 {{is a protected member of}}
// expected-error@+1 {{is a private member of}}
template <typename T> struct PCTT1<T, TestClass::AliasPrivateClass> final : PCTT2<T, TestClass::ProtectedClass, TestClass::PrivateClass> {};
// expected-error@+1 {{is a protected member of}}
template <typename T> class PCTT2<T, TestClass::PrivateClass, TestClass::TemplatePrivateClass<T>> : PCTT3<T, TestClass::protectedStaticInt> {};
// expected-error@+1 {{is a protected member of}}
template <typename T> class PCTT3<T, TestClass::privateStaticInt> final : PCTT4<T, &TestClass::protectedFunc> {};
// expected-error@+1 {{is a protected member of}}
template <typename T> struct PCTT4<T, &TestClass::privateFunc> : PCTT5<T, &TestClass::protectedStaticFunc> {};
// expected-error@+1 {{is a protected member of}}
template <typename T> class PCTT4<T, &TestClass::privateFuncOverloaded> final : PCTT5<T, &TestClass::protectedStaticFuncOverloaded> {};
template <typename T> class PCTT5<T, &TestClass::privateStaticFunc> : P1 {};
// expected-error@+2 {{cannot specialize a dependent template}}
// expected-error@+1 {{is a private member of}}
template <typename T> class PCTT6<T, TestClass::PrivateClass>::template PCTT1<TestClass::PrivateClass> : PCTT6<T, TestClass::PrivateClass>::template NCT2<int> {};
// expected-error@+1 {{is a private member of}}
template <typename T> class PCTT5<T, &TestClass::privateStaticFuncOverloaded> final : PCTT6<T, T>::template NCT1<TestClass::PrivateClass> {};
template <typename T> class PCTT6<TestClass::PrivateClass, T> {
  template <typename T1, typename T2> class NCT3 final {};
  template <typename T1, typename T2> class NCT4;
};
template <typename T1> template <typename, typename> class PCTT6<TestClass::PrivateClass, T1>::NCT4 final {};
// expected-error@+1 2{{is a private member of}}
template <typename T1> template <typename T2> struct PCTT6<TestClass::PrivateClass, T1>::NCT3<T2, TestClass::TemplatePrivateClass<TestClass::TemplateProtectedClass<TestClass::PublicClass>>> : PCTT6<TestClass::PrivateClass, T1>::NCT4<T2, TestClass::TemplatePrivateClass<int>> {};
