// RUN: %clang_cc1 -fms-compatibility -std=c++11 %s -verify

// MSVC should compile this file without errors.

namespace test_basic {
template <typename T = Baz> // expected-warning {{using the undeclared type 'Baz' as a default template argument is a Microsoft extension}}
struct Foo { T x; };
typedef int Baz;
template struct Foo<>;
}

namespace test_namespace {
namespace nested {
template <typename T = Baz> // expected-warning {{using the undeclared type 'Baz' as a default template argument is a Microsoft extension}}
struct Foo {
  static_assert(sizeof(T) == 4, "should get int, not double");
};
typedef int Baz;
}
typedef double Baz;
template struct nested::Foo<>;
}

namespace test_inner_class_template {
struct Outer {
  template <typename T = Baz> // expected-warning {{using the undeclared type 'Baz' as a default template argument is a Microsoft extension}}
  struct Foo {
    static_assert(sizeof(T) == 4, "should get int, not double");
  };
  typedef int Baz;
};
typedef double Baz;
template struct Outer::Foo<>;
}

namespace test_nontype_param {
template <typename T> struct Bar { T x; };
typedef int Qux;
template <Bar<Qux> *P>
struct Foo {
};
Bar<int> g;
template struct Foo<&g>;
}

// MSVC accepts this, but Clang doesn't.
namespace test_template_instantiation_arg {
template <typename T> struct Bar { T x; };
template <typename T = Bar<Weber>>  // expected-error {{use of undeclared identifier 'Weber'}}
struct Foo {
  static_assert(sizeof(T) == 4, "Bar should have gotten int");
  // FIXME: These diagnostics are bad.
}; // expected-error {{expected ',' or '>' in template-parameter-list}}
// expected-warning@-1 {{does not declare anything}}
typedef int Weber;
}

// MSVC accepts this, but Clang doesn't.
namespace test_scope_spec {
template <typename T = ns::Bar>  // expected-error {{use of undeclared identifier 'ns'}}
struct Foo {
  static_assert(sizeof(T) == 4, "Bar should have gotten int");
};
namespace ns { typedef int Bar; }
}

#ifdef __clang__
// These are negative test cases that MSVC doesn't compile either.  Try to use
// unique undeclared identifiers so typo correction doesn't find types declared
// above.

namespace test_undeclared_nontype_parm_type {
template <Zargon N> // expected-error {{unknown type name 'Zargon'}}
struct Foo { int x[N]; };
typedef int Zargon;
template struct Foo<4>;
}

namespace test_undeclared_nontype_parm_type_no_name {
template <typename T, Asdf> // expected-error {{unknown type name 'Asdf'}}
struct Foo { T x; };
template struct Foo<int, 0>;
}

namespace test_undeclared_type_arg {
template <typename T>
struct Foo { T x; };
template struct Foo<Yodel>; // expected-error {{use of undeclared identifier 'Yodel'}}
}

namespace test_undeclared_nontype_parm_arg {
// Bury an undeclared type as a template argument to the type of a non-type
// template parameter.
template <typename T> struct Bar { T x; };

template <Bar<Xylophone> *P> // expected-error {{use of undeclared identifier 'Xylophone'}}
// expected-note@-1 {{template parameter is declared here}}
struct Foo { };

typedef int Xylophone;
Bar<Xylophone> g;
template struct Foo<&g>; // expected-error {{cannot be converted}}
}

#endif
