// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s

namespace test1 {
  template<typename T> struct Base {
    template<typename V = T> requires true
    Base(T);
  };

  template<typename T> struct InheritsCtors : public Base<T> {
    using Base<T>::Base;
  };

  InheritsCtors inheritsCtors(1);
  using IC = InheritsCtors<int>;
  using IC = decltype(inheritsCtors);

  template<typename T> struct DoesNotInheritCtors : public Base<T> {}; // expected-note {{candidate template ignored: could not match 'DoesNotInheritCtors<T>' against 'int'}} \
                                                                       // expected-note 3{{implicit deduction guide declared as}} \
                                                                       // expected-note {{candidate function template not viable: requires 0 arguments, but 1 was provided}} \
                                                                       // expected-note {{candidate template ignored: could not match 'Base<T>' against 'int'}}
  DoesNotInheritCtors doesNotInheritCtors(100); // expected-error {{no viable constructor or deduction guide for deduction of template arguments}}

  template<typename T> struct InheritsSecond : public Base<T> {
    using Base<T>::Base;
  };

  InheritsSecond inheritsSecond('a');
  using IS = InheritsSecond<char>;
  using IS = decltype(inheritsSecond);

  template<typename T> struct NonTemplateDGuideBase {
    NonTemplateDGuideBase(T); // expected-note {{generated from 'NonTemplateDGuideBase<T>' constructor}}
  };
  NonTemplateDGuideBase(int) -> NonTemplateDGuideBase<char>;
  NonTemplateDGuideBase(const char *) -> NonTemplateDGuideBase<const char *>;

  template<typename T>
  concept NoPointers = !requires (T t) { *t; };

  template<NoPointers T>
  struct NonTemplateDGuideDerived : public NonTemplateDGuideBase<T> { // expected-note {{candidate template ignored: could not match 'NonTemplateDGuideDerived<T>' against 'const char *'}} \
                                                                      // expected-note {{candidate function template not viable: requires 0 arguments, but 1 was provided}} \
                                                                      // expected-note 2{{implicit deduction guide declared as }}
    using NonTemplateDGuideBase<T>::NonTemplateDGuideBase; // expected-note {{candidate function not viable: no known conversion from 'const char[1]' to 'int' for 1st argument}} \
                                                           // expected-note {{candidate template ignored: could not deduce template arguments for 'NonTemplateDGuideDerived<T>' from 'NonTemplateDGuideBase<T>' [with T = const char *]}} \
                                                           // expected-note {{implicit deduction guide declared as 'template <NoPointers<> T> requires __is_deducible(test1::NonTemplateDGuideDerived, typename __ctad_mapper_NonTemplateDGuideBase_to_NonTemplateDGuideDerived_0<NonTemplateDGuideBase<T>>::type) NonTemplateDGuideDerived(T) -> typename __ctad_mapper_NonTemplateDGuideBase_to_NonTemplateDGuideDerived_0<NonTemplateDGuideBase<T>>::type'}} \
                                                           // expected-note {{candidate template ignored: could not match 'NonTemplateDGuideBase<T>' against 'const char *'}} \
                                                           // expected-note {{implicit deduction guide declared as 'template <NoPointers<> T> requires __is_deducible(test1::NonTemplateDGuideDerived, typename __ctad_mapper_NonTemplateDGuideBase_to_NonTemplateDGuideDerived_0<NonTemplateDGuideBase<T>>::type) NonTemplateDGuideDerived(NonTemplateDGuideBase<T>) -> typename __ctad_mapper_NonTemplateDGuideBase_to_NonTemplateDGuideDerived_0<NonTemplateDGuideBase<T>>::type'}}
  };

  NonTemplateDGuideDerived ntdg(1);
  using NTDG = NonTemplateDGuideDerived<char>;
  using NTDG = decltype(ntdg);

  NonTemplateDGuideDerived ntdg_char(""); // expected-error {{no viable constructor or deduction guide for deduction of template arguments}}

  template<typename T>
  struct ExplicitBase {
    template<typename V>
    ExplicitBase(V);
  };

  template<typename T>
  ExplicitBase(T) -> ExplicitBase<T>;

  template<NoPointers T>
  struct ExplicitDerived : public ExplicitBase<T> { // expected-note {{candidate template ignored: could not match 'ExplicitDerived<T>' against 'const char *'}} \
                                                    // expected-note {{candidate function template not viable: requires 0 arguments, but 1 was provided}} \
                                                    // expected-note 2{{implicit deduction guide declared as }}

    using ExplicitBase<T>::ExplicitBase; // expected-note {{candidate template ignored: couldn't infer template argument 'T'}} \
                                         // expected-note {{implicit deduction guide declared as 'template <NoPointers<> T, typename V> requires __is_deducible(test1::ExplicitDerived, typename __ctad_mapper_ExplicitBase_to_ExplicitDerived_0<ExplicitBase<T>>::type) ExplicitDerived(V) -> typename __ctad_mapper_ExplicitBase_to_ExplicitDerived_0<ExplicitBase<T>>::type'}} \
                                         // expected-note {{candidate template ignored: could not match 'ExplicitBase<T>' against 'const char *'}} \
                                         // expected-note {{candidate template ignored: could not deduce template arguments for 'ExplicitDerived<T>' from 'ExplicitBase<T>' [with T = const char *]}} \
                                         // expected-note {{implicit deduction guide declared as 'template <NoPointers<> T> requires __is_deducible(test1::ExplicitDerived, typename __ctad_mapper_ExplicitBase_to_ExplicitDerived_0<ExplicitBase<T>>::type) ExplicitDerived(ExplicitBase<T>) -> typename __ctad_mapper_ExplicitBase_to_ExplicitDerived_0<ExplicitBase<T>>::type'}}
  };

  ExplicitDerived ed(10);
  using ED = ExplicitDerived<int>;
  using ED = decltype(ed);

  ExplicitDerived substFail(""); // expected-error {{no viable constructor or deduction guide for deduction of template arguments}}

  // FIXME: Support deduction guides that were declared
  // after the initial implicit guides are declared for
  // the derived template.
#if 0
  Base(int) -> Base<char>;

  InheritsCtors ic2(1);
  using IC2 = InheritsCtors<char>;
  using IC2 = decltype(ic2);
#endif
}

namespace test2 {
  template<typename T, typename U, typename V> struct Base {
    Base(T, U, V);
  };

  template<typename T, typename U> struct Derived : public Base<U, T, int> {
    // FIXME: Current implementation does not find Base's constructors correctly
    // with the below using decl
    //using Derived::Base::Base;

    using Base<U, T, int>::Base;
  };

  Derived derived(true, 'a', 1);
  using D = Derived<char, bool>;
  using D = decltype(derived);
}

namespace test3 {
  template<typename T> struct Base {
    Base(T); // expected-note {{generated from 'Base<T>' constructor}}
  };

  template<typename T, typename U> struct NotEnoughParams : public Base<T> { // expected-note {{candidate template ignored: could not match 'NotEnoughParams<T, U>' against 'int'}} \
                                                                             // expected-note {{candidate template ignored: could not match 'Base<T>' against 'int'}} \
                                                                             // expected-note {{candidate function template not viable: requires 0 arguments, but 1 was provided}} \
                                                                             // expected-note 3{{implicit deduction guide declared as}}
    using Base<T>::Base; // expected-note {{candidate template ignored: could not deduce template arguments for 'NotEnoughParams<T, U>' from 'Base<T>' [with T = int]}} \
                         // expected-note {{implicit deduction guide declared as 'template <typename T> requires __is_deducible(test3::NotEnoughParams, typename __ctad_mapper_Base_to_NotEnoughParams_0<Base<T>>::type) NotEnoughParams(T) -> typename __ctad_mapper_Base_to_NotEnoughParams_0<Base<T>>::type'}} \
                         // expected-note {{candidate template ignored: could not match 'Base<T>' against 'int'}} \
                         // expected-note {{implicit deduction guide declared as 'template <typename T> requires __is_deducible(test3::NotEnoughParams, typename __ctad_mapper_Base_to_NotEnoughParams_0<Base<T>>::type) NotEnoughParams(Base<T>) -> typename __ctad_mapper_Base_to_NotEnoughParams_0<Base<T>>::type'}}
  };

  NotEnoughParams notEnoughParams(1); // expected-error {{no viable constructor or deduction guide for deduction of template arguments}}
}

namespace test4 {
  template<typename T> struct B {
    B(T t);
  };

  template<typename T = int, typename V = int>
  struct DefaultArgsNotInBase : public B<V> {
    using B<V>::B;
  };

  DefaultArgsNotInBase d('d');
  using D = DefaultArgsNotInBase<int, char>;
  using D = decltype(d);

  template<typename T> struct BaseEmptyCtor {
    BaseEmptyCtor();
  };

  template<typename T = int, typename V = int>
  struct DefaultArgsNotInBaseEmpty : public BaseEmptyCtor<V> {
    using BaseEmptyCtor<V>::BaseEmptyCtor;
  };

  DefaultArgsNotInBaseEmpty d2;
  using D2 = DefaultArgsNotInBaseEmpty<>;
  using D2 = decltype(d2);
}

namespace test5 {
  template<typename T> struct Base {
    Base(T);
  };
  template<typename T> struct Outer {
    template<typename U> struct Inner : public Base<U> {
      using Base<U>::Base;
    };
  };

  Outer<int>::Inner i(10);
  using I = Outer<int>::Inner<int>;
  using I = decltype(i);
}

namespace test6 {
  template<typename T>
    concept False = false;

  template<typename T>
    concept True = true;

  template<typename T>
  struct Base {
    Base(T); // expected-note {{generated from 'Base<T>' constructor}}
  };

  template<False F>
  struct DerivedFalse : public Base<F> { // expected-note {{candidate template ignored: could not match 'DerivedFalse<F>' against 'int'}} \
                                         // expected-note {{candidate template ignored: could not match 'Base<F>' against 'int'}} \
                                         // expected-note {{candidate function template not viable: requires 0 arguments, but 1 was provided}} \
                                         // expected-note 3{{implicit deduction guide declared as}}
      using Base<F>::Base; // expected-note {{candidate template ignored: could not deduce template arguments for 'DerivedFalse<F>' from 'Base<T>' [with F = int]}} \
                           // expected-note {{implicit deduction guide declared as 'template <False<> F> requires __is_deducible(test6::DerivedFalse, typename __ctad_mapper_Base_to_DerivedFalse_0<Base<F>>::type) DerivedFalse(F) -> typename __ctad_mapper_Base_to_DerivedFalse_0<Base<F>>::type'}} \
                           // expected-note {{candidate template ignored: could not match 'Base<F>' against 'int'}} \
                           // expected-note {{implicit deduction guide declared as 'template <False<> F> requires __is_deducible(test6::DerivedFalse, typename __ctad_mapper_Base_to_DerivedFalse_0<Base<F>>::type) DerivedFalse(Base<F>) -> typename __ctad_mapper_Base_to_DerivedFalse_0<Base<F>>::type'}}
    };

  template<True F>
  struct DerivedTrue : public Base<F> {
    using Base<F>::Base;
  };

  DerivedFalse df(10); // expected-error {{no viable constructor or deduction guide for deduction of template arguments}}

  DerivedTrue dt(10);
  using DT = DerivedTrue<int>;
  using DT = decltype(dt);
}

namespace test7 {
  template<typename T, typename U>
  struct Base1 {
    Base1();
    Base1(T, U);
  };

  template<typename T, typename U>
  struct Base2 {
    Base2();
    Base2(T, U);
  };

  template<typename T = int, typename U = int>
  struct MultipleInheritance : public Base1<T, U*> , Base2<U*, T> { 
    using Base1<T, U*>::Base1; // expected-note {{candidate function [with T = int, U = int]}}
    using Base2<U*, T>::Base2; // expected-note {{candidate function [with T = int, U = int]}}
  };

  MultipleInheritance mi1(1, "");
  using MI1 = MultipleInheritance<int, const char>;
  using MI1 = decltype(mi1);

  MultipleInheritance mi2("", 1);
  using MI2 = MultipleInheritance<int, const char>;
  using MI2 = decltype(mi2);

  // This is an odd case.
  // Since the base DGs have the deducible constraint, they are more specialized than MultipleInheritance's
  // ctor, and take priority before the new clause in P2582R1 is applied.
  MultipleInheritance mi3; // expected-error {{ambiguous deduction for template arguments of 'MultipleInheritance'}}

  template<typename T>
  struct MultipleInheritanceSameBase : public Base1<T, const T*>, Base1<const T*, T> {
    using Base1<T, const T*>::Base1;
    using Base1<const T*, T>::Base1;
  };

  MultipleInheritanceSameBase misb1('a', "");
  using MISB1 = MultipleInheritanceSameBase<char>;
  using MISB1 = decltype(misb1);

  MultipleInheritanceSameBase misb2("", 'a');
  using MISB2 = MultipleInheritanceSameBase<char>;
  using MISB2 = decltype(misb2);
}

namespace test8 {
  template<typename T>
  struct Base {
    Base(T);
  };

  template<typename T>
  struct SpecializedBase : Base<int> {
    using Base<T>::Base; // expected-error {{using declaration refers into 'Base<char>::', which is not a base class of 'SpecializedBase<char>'}}
  };

  SpecializedBase sb1(10);
  static_assert(__is_same(decltype(sb1), SpecializedBase<int>));

  SpecializedBase sb2('a'); // expected-note {{in instantiation of template class 'test8::SpecializedBase<char>' requested here}}
}
