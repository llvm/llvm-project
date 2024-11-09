// RUN: %clang_cc1 -fsyntax-only -triple arm64-apple-ios   -verify -fptrauth-calls -std=c++2a %s
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-linux-gnu -verify -fptrauth-calls -std=c++2a %s

namespace basic {

#define authenticated(a, b, c...) [[clang::ptrauth_vtable_pointer(a, b, c)]]

// Basic sanity tests
#define TEST_AUTH(name, auth...)                        \
  struct [[clang::ptrauth_vtable_pointer(auth)]] name { \
    virtual ~name() {}                                  \
  }

TEST_AUTH(NoParams);
// expected-error@-1{{'ptrauth_vtable_pointer' attribute takes at least 3 arguments}}
TEST_AUTH(NoAuth, no_authentication, default_address_discrimination, default_extra_discrimination);
TEST_AUTH(InvalidKey, wat, default_address_discrimination, default_extra_discrimination);
// expected-error@-1{{invalid authentication key 'wat'}}
TEST_AUTH(InvalidAddressDiscrimination, no_authentication, wat, default_extra_discrimination);
// expected-error@-1{{invalid address discrimination mode 'wat'}}
TEST_AUTH(InvalidExtraDiscrimination, no_authentication, default_address_discrimination, wat);
// expected-error@-1{{invalid extra discrimination selection 'wat'}}
TEST_AUTH(InvalidNoCustomDiscrimination, no_authentication, default_address_discrimination, custom_discrimination);
// expected-error@-1{{missing custom discrimination}}
TEST_AUTH(InvalidCustomDiscrimination, no_authentication, default_address_discrimination, custom_discrimination, wat);
// expected-error@-1{{invalid custom discrimination}}
TEST_AUTH(Default, default_key, default_address_discrimination, default_extra_discrimination);
TEST_AUTH(InvalidDefaultExtra, default_key, default_address_discrimination, default_extra_discrimination, 1);
// expected-error@-1{{'ptrauth_vtable_pointer' attribute takes no more than 3 arguments}}
TEST_AUTH(ProcessDependentKey, process_dependent, default_address_discrimination, default_extra_discrimination);
TEST_AUTH(ProcessIndependentKey, process_independent, default_address_discrimination, default_extra_discrimination);
TEST_AUTH(DefaultAddressDiscrimination, process_independent, default_address_discrimination, default_extra_discrimination);
TEST_AUTH(NoAddressDiscrimination, process_independent, no_address_discrimination, default_extra_discrimination);
TEST_AUTH(AddressDiscrimination, process_independent, address_discrimination, default_extra_discrimination);
TEST_AUTH(DefaultExtraDiscrimination, process_independent, default_address_discrimination, default_extra_discrimination);
TEST_AUTH(NoExtraDiscrimination, process_independent, default_address_discrimination, no_extra_discrimination);
TEST_AUTH(TypeExtraDiscrimination, process_independent, default_address_discrimination, type_discrimination);
TEST_AUTH(InvalidCustomExtraDiscrimination, process_independent, default_address_discrimination, custom_discrimination);
// expected-error@-1{{missing custom discrimination}}
TEST_AUTH(ValidCustomExtraDiscrimination, process_independent, default_address_discrimination, custom_discrimination, 1);

// Basic valid authentication configuration
#define generic_authenticated \
  authenticated(process_independent, address_discrimination, type_discrimination)

struct generic_authenticated ForwardDecl;

struct generic_authenticated generic_authenticated InvalidDuplicateAttribute {
  // expected-error@-1{{multiple vtable pointer authentication policies on 'InvalidDuplicateAttribute'}}
  virtual ~InvalidDuplicateAttribute(){};
};
struct generic_authenticated ValidPolymorphic {
  virtual ~ValidPolymorphic(){};
};
struct generic_authenticated InvalidMonomorphic { // expected-error{{cannot set vtable pointer authentication on monomorphic type 'InvalidMonomorphic'}}
};
struct ValidMonomorphic {
};

struct ValidSubclass : ValidPolymorphic {};
struct generic_authenticated InvalidSubclass : ValidPolymorphic {}; // expected-error{{cannot set vtable pointer authentication on 'InvalidSubclass' which is a subclass of polymorphic type 'ValidPolymorphic'}}

// Awful template time
template <typename T>
struct generic_authenticated ExplicitlyAuthedMonomorphicTemplateClass : T {};
// expected-error@-1{{cannot set vtable pointer authentication on 'ExplicitlyAuthedMonomorphicTemplateClass<basic::ValidPolymorphic>' which is a subclass of polymorphic type 'ValidPolymorphic'}}
// expected-error@-2{{cannot set vtable pointer authentication on monomorphic type 'ExplicitlyAuthedMonomorphicTemplateClass<basic::ValidMonomorphic>'}}
template <typename T>
struct generic_authenticated ExplicitlyAuthedPolymorphicTemplateClass : T { // expected-error{{cannot set vtable pointer authentication on 'ExplicitlyAuthedPolymorphicTemplateClass<basic::ValidPolymorphic>' which is a subclass of polymorphic type 'ValidPolymorphic'}}
  virtual ~ExplicitlyAuthedPolymorphicTemplateClass(){};
};
template <typename T>
struct UnauthedMonomorphicTemplateClass : T {};
template <typename T>
struct UnauthedPolymorphicTemplateClass : T {
  virtual ~UnauthedPolymorphicTemplateClass(){};
};

ExplicitlyAuthedMonomorphicTemplateClass<ValidPolymorphic> test1;
// expected-note@-1{{in instantiation of template class 'basic::ExplicitlyAuthedMonomorphicTemplateClass<basic::ValidPolymorphic>' requested here}}
ExplicitlyAuthedMonomorphicTemplateClass<ValidMonomorphic> test2;
// expected-note@-1{{in instantiation of template class 'basic::ExplicitlyAuthedMonomorphicTemplateClass<basic::ValidMonomorphic>' requested here}}
ExplicitlyAuthedPolymorphicTemplateClass<ValidPolymorphic> test3;
// expected-note@-1{{in instantiation of template class 'basic::ExplicitlyAuthedPolymorphicTemplateClass<basic::ValidPolymorphic>' requested here}}
ExplicitlyAuthedPolymorphicTemplateClass<ValidMonomorphic> test4;

UnauthedMonomorphicTemplateClass<ValidPolymorphic> test5;
UnauthedMonomorphicTemplateClass<ValidMonomorphic> test6;
UnauthedPolymorphicTemplateClass<ValidPolymorphic> test7;
UnauthedPolymorphicTemplateClass<ValidMonomorphic> test8;

// Just use a different policy from the generic macro to verify we won't complain
// about the insanity
struct authenticated(process_independent, no_address_discrimination, type_discrimination) SecondAuthenticatedPolymorphic {
  virtual ~SecondAuthenticatedPolymorphic(){};
};
struct UnauthenticatedPolymorphic {
  virtual ~UnauthenticatedPolymorphic(){};
};

struct MultipleParents1 : ValidPolymorphic, SecondAuthenticatedPolymorphic, UnauthenticatedPolymorphic {};
struct MultipleParents2 : UnauthenticatedPolymorphic, ValidPolymorphic, SecondAuthenticatedPolymorphic {};
struct generic_authenticated InvalidMultipleParents : UnauthenticatedPolymorphic, ValidPolymorphic, SecondAuthenticatedPolymorphic {};
// expected-error@-1{{cannot set vtable pointer authentication on 'InvalidMultipleParents' which is a subclass of polymorphic type 'UnauthenticatedPolymorphic'}}

template <typename T>
struct generic_authenticated ExplicitlyAuthedPolymorphicTemplateClassNoBase {
  virtual ~ExplicitlyAuthedPolymorphicTemplateClassNoBase();
};

ExplicitlyAuthedPolymorphicTemplateClassNoBase<int> v;

struct ValidSubclassOfTemplate : ExplicitlyAuthedPolymorphicTemplateClassNoBase<int> {
};

struct generic_authenticated InvalidSubclassOfTemplate : ExplicitlyAuthedPolymorphicTemplateClassNoBase<int> {
  // expected-error@-1{{cannot set vtable pointer authentication on 'InvalidSubclassOfTemplate' which is a subclass of polymorphic type 'ExplicitlyAuthedPolymorphicTemplateClassNoBase<int>'}}
};

template <typename T>
struct generic_authenticated ExplicitlyAuthedMonomorphicTemplateClassNoBase {
  // expected-error@-1{{cannot set vtable pointer authentication on monomorphic type 'ExplicitlyAuthedMonomorphicTemplateClassNoBase'}}
  // expected-error@-2{{cannot set vtable pointer authentication on monomorphic type 'ExplicitlyAuthedMonomorphicTemplateClassNoBase<int>'}}
};

ExplicitlyAuthedMonomorphicTemplateClassNoBase<int> X;
// expected-note@-1{{in instantiation of template class 'basic::ExplicitlyAuthedMonomorphicTemplateClassNoBase<int>' requested here}}

template <typename T>
struct generic_authenticated ExplicitlyAuthedTemplateClassValidBase : ValidMonomorphic {
  // expected-error@-1{{cannot set vtable pointer authentication on monomorphic type 'ExplicitlyAuthedTemplateClassValidBase'}}
  // expected-error@-2{{cannot set vtable pointer authentication on monomorphic type 'ExplicitlyAuthedTemplateClassValidBase<int>'}}
};

ExplicitlyAuthedTemplateClassValidBase<int> Y;
// expected-note@-1{{in instantiation of template class 'basic::ExplicitlyAuthedTemplateClassValidBase<int>' requested here}}

template <typename T>
struct generic_authenticated ExplicitlyAuthedTemplateClassInvalidBase : ValidPolymorphic {
  // expected-error@-1{{cannot set vtable pointer authentication on 'ExplicitlyAuthedTemplateClassInvalidBase' which is a subclass of polymorphic type 'ValidPolymorphic'}}
  // expected-error@-2{{cannot set vtable pointer authentication on 'ExplicitlyAuthedTemplateClassInvalidBase<int>' which is a subclass of polymorphic type 'ValidPolymorphic'}}
};

ExplicitlyAuthedTemplateClassInvalidBase<int> Z;
// expected-note@-1{{in instantiation of template class 'basic::ExplicitlyAuthedTemplateClassInvalidBase<int>' requested here}}

template <class test1, class test2>
class generic_authenticated TestPolymorphicTemplateSpecialization;

template <>
class TestPolymorphicTemplateSpecialization<double, float> {
  MissingDecl *zl;
  // expected-error@-1 {{unknown type name 'MissingDecl'}}
public:
  virtual ~TestPolymorphicTemplateSpecialization();
};
template <class test1>
class generic_authenticated TestPolymorphicTemplateSpecialization<test1, double>
// expected-error@-1 {{cannot set vtable pointer authentication on monomorphic type 'TestPolymorphicTemplateSpecialization<test1, double>'}}
// expected-error@-2 {{cannot set vtable pointer authentication on monomorphic type 'TestPolymorphicTemplateSpecialization<double, double>'}}
{
};

TestPolymorphicTemplateSpecialization<double, float> b;
TestPolymorphicTemplateSpecialization<double, double> b2;
// expected-note@-1 {{in instantiation of template class 'basic::TestPolymorphicTemplateSpecialization<double, double>' requested here}}

template <typename A> class generic_authenticated TestMonomorphic {};
// expected-error@-1 {{cannot set vtable pointer authentication on monomorphic type 'TestMonomorphic'}}
// expected-error@-2 {{cannot set vtable pointer authentication on monomorphic type 'TestMonomorphic<double>'}}

template <> class generic_authenticated TestMonomorphic<int> {
public:
  virtual ~TestMonomorphic();
};

struct TestMonomorphicSubclass : TestMonomorphic<int> {
};
template <typename T> struct generic_authenticated TestMonomorphicSubclass2 : TestMonomorphic<T> {
  // expected-error@-1 {{cannot set vtable pointer authentication on 'TestMonomorphicSubclass2<int>' which is a subclass of polymorphic type 'TestMonomorphic<int>'}}
  // expected-error@-2 {{cannot set vtable pointer authentication on monomorphic type 'TestMonomorphicSubclass2<double>'}}
  // expected-note@-3 {{in instantiation of template class 'basic::TestMonomorphic<double>' requested here}}
};

TestMonomorphicSubclass tms_1;
TestMonomorphicSubclass2<int> tms2_1;
// expected-note@-1 {{in instantiation of template class 'basic::TestMonomorphicSubclass2<int>' requested here}}
TestMonomorphicSubclass2<double> tms2_2;
// expected-note@-1 {{in instantiation of template class 'basic::TestMonomorphicSubclass2<double>' requested here}}
// expected-note@-2 {{in instantiation of template class 'basic::TestMonomorphicSubclass2<double>' requested here}}

template <typename T>
class generic_authenticated dependent_type {
  // expected-error@-1 {{cannot set vtable pointer authentication on monomorphic type 'dependent_type'}}
  static constexpr unsigned small_object_size = 1;
  char _model[small_object_size];
};

template <typename... T>
class generic_authenticated dependent_type2 : public T... {
  // expected-error@-1 {{cannot set vtable pointer authentication on 'dependent_type2<basic::Foo>' which is a subclass of polymorphic type 'Foo'}}
  static constexpr unsigned small_object_size = 1;
  char _model[small_object_size];
};

struct Foo {
  virtual ~Foo();
};

dependent_type2<Foo> thing;
// expected-note@-1 {{in instantiation of template class 'basic::dependent_type2<basic::Foo>' requested here}}

template <class>
class task;
template <unsigned align> struct alignedthing {
  char buffer[align];
};

template <class R, class... Args>
class generic_authenticated task<R(Args...)> {
  // expected-error@-1 {{cannot set vtable pointer authentication on monomorphic type 'task<R (Args...)>'}}
  static constexpr __SIZE_TYPE__ small_object_size = 256;
  alignedthing<small_object_size> _model;
};

} // namespace basic
