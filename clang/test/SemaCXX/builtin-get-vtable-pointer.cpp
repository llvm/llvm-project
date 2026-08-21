// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s

namespace basic {
struct ForwardDeclaration; // expected-note{{forward declaration of 'basic::ForwardDeclaration'}}
                           // expected-note@-1{{forward declaration of 'basic::ForwardDeclaration'}}
struct NonPolymorphic {};
struct Polymorphic {
  virtual ~Polymorphic();
};

template <typename T>
struct Foo {
  virtual ~Foo();
};

template <>
struct Foo<int> {
};

template <typename T>
struct Bar {
  using SubType = typename T::SubType;
  SubType *ty() const;
};

struct Thing1 {
  using SubType = Thing1;
};

struct Thing2 {
  using SubType = Thing2;
  virtual ~Thing2();
};

struct Thing3 {
  using SubType = int;
};

struct Thing4 {
  using SubType = Polymorphic;
};

struct Thing5 {
  using SubType = NonPolymorphic;
};

struct Thing6 {
  using SubType = ForwardDeclaration;
};

template <typename T>
const void *getThing(const Bar<T> *b = nullptr) {
  return __builtin_get_vtable_pointer(b->ty()); // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'SubType *' (aka 'int *') was provided}}
                                                // expected-error@-1{{__builtin_get_vtable_pointer requires an argument of polymorphic class pointer type, but 'Thing1' has no virtual methods}}
                                                // expected-error@-2{{__builtin_get_vtable_pointer requires an argument of polymorphic class pointer type, but 'NonPolymorphic' has no virtual methods}}
                                                // expected-error@-3{{__builtin_get_vtable_pointer requires an argument with a complete type, but 'SubType' (aka 'ForwardDeclaration') is incomplete}}
}
template <typename>
struct IncompleteTemplate; // expected-note{{template is declared here}}
template <typename>
struct MonomorphicTemplate {
};
template <typename>
struct PolymorphicTemplate {
  virtual ~PolymorphicTemplate();
};

void test_function(int);    // expected-note{{possible target for call}}
void test_function(double); // expected-note{{possible target for call}}

void getVTablePointer() {
  ForwardDeclaration *fd = nullptr;
  NonPolymorphic np;
  Polymorphic p;
  NonPolymorphic np_array[1];
  Polymorphic p_array[1];
  __builtin_get_vtable_pointer(0);             // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'int' was provided}}
  __builtin_get_vtable_pointer(nullptr);       // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'std::nullptr_t' was provided}}
  __builtin_get_vtable_pointer(0.5);           // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'double' was provided}}
  __builtin_get_vtable_pointer(fd);            // expected-error{{__builtin_get_vtable_pointer requires an argument with a complete type, but 'ForwardDeclaration' is incomplete}}
  __builtin_get_vtable_pointer(np);            // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'NonPolymorphic' was provided}}
  __builtin_get_vtable_pointer(&np);           // expected-error{{__builtin_get_vtable_pointer requires an argument of polymorphic class pointer type, but 'NonPolymorphic' has no virtual methods}}
  __builtin_get_vtable_pointer(p);             // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'Polymorphic' was provided}}
  __builtin_get_vtable_pointer(&p);            // expected-warning{{ignoring return value of function declared with const attribute}}
  __builtin_get_vtable_pointer(p_array);       // expected-warning{{ignoring return value of function declared with const attribute}}
  __builtin_get_vtable_pointer(&p_array);      // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'Polymorphic (*)[1]' was provided}}
  __builtin_get_vtable_pointer(np_array);      // expected-error{{__builtin_get_vtable_pointer requires an argument of polymorphic class pointer type, but 'NonPolymorphic' has no virtual methods}}
  __builtin_get_vtable_pointer(&np_array);     // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'NonPolymorphic (*)[1]' was provided}}
  __builtin_get_vtable_pointer(test_function); // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
  Foo<double> Food;
  Foo<int> Fooi;
  __builtin_get_vtable_pointer(Food); // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'Foo<double>' was provided}}
  (void)__builtin_get_vtable_pointer(&Food);
  __builtin_get_vtable_pointer(Fooi);  // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'Foo<int>' was provided}}
  __builtin_get_vtable_pointer(&Fooi); // expected-error{{__builtin_get_vtable_pointer requires an argument of polymorphic class pointer type, but 'Foo<int>' has no virtual methods}}

  IncompleteTemplate<bool> *incomplete = nullptr;
  (void)__builtin_get_vtable_pointer(incomplete); // expected-error{{implicit instantiation of undefined template 'basic::IncompleteTemplate<bool>'}}
  PolymorphicTemplate<bool> *ptb = nullptr;
  MonomorphicTemplate<bool> *mtb = nullptr;
  PolymorphicTemplate<int> pti;
  MonomorphicTemplate<int> mti;
  PolymorphicTemplate<float> ptf;
  MonomorphicTemplate<float> mtf;
  (void)__builtin_get_vtable_pointer(ptb);
  __builtin_get_vtable_pointer(mtb); // expected-error{{__builtin_get_vtable_pointer requires an argument of polymorphic class pointer type, but 'MonomorphicTemplate<bool>' has no virtual methods}}
  __builtin_get_vtable_pointer(pti); // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'PolymorphicTemplate<int>' was provided}}
  __builtin_get_vtable_pointer(mti); // expected-error{{__builtin_get_vtable_pointer requires an argument of class pointer type, but 'MonomorphicTemplate<int>' was provided}}
  (void)__builtin_get_vtable_pointer(&ptf);
  __builtin_get_vtable_pointer(&mtf); // expected-error{{__builtin_get_vtable_pointer requires an argument of polymorphic class pointer type, but 'MonomorphicTemplate<float>' has no virtual methods}}

  getThing<Thing1>(); // expected-note{{in instantiation of function template specialization 'basic::getThing<basic::Thing1>' requested here}}
  getThing<Thing2>();
  getThing<Thing3>(); // expected-note{{in instantiation of function template specialization 'basic::getThing<basic::Thing3>' requested here}}
  getThing<Thing4>();
  getThing<Thing5>(); // expected-note{{in instantiation of function template specialization 'basic::getThing<basic::Thing5>' requested here}}
  getThing<Thing6>(); // expected-note{{in instantiation of function template specialization 'basic::getThing<basic::Thing6>' requested here}}
}

} // namespace basic
