// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-mingw                 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-cygwin                -fsyntax-only -verify %s

// Test that memberwise dllexport and dllimport are warned if the
// exclude_from_explicit_instantiation attribute is attached.

#define EXCLUDE_ATTR __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct C {
  EXCLUDE_ATTR __declspec(dllexport) void fn_excluded_exported(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  EXCLUDE_ATTR __declspec(dllimport) void fn_excluded_imported(); // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  EXCLUDE_ATTR __declspec(dllexport) static int var_excluded_exported; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  EXCLUDE_ATTR __declspec(dllimport) static int var_excluded_imported; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  struct EXCLUDE_ATTR __declspec(dllexport) nested_excluded_exported {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  struct EXCLUDE_ATTR __declspec(dllimport) nested_excluded_imported {}; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

  // No warnings here since nested_excluded is not instantiated.
  struct EXCLUDE_ATTR nested_excluded {
    __declspec(dllexport) void fn_exported();
    __declspec(dllimport) void fn_imported();
  };
  // This too. nested_exported is not instantiated.
  struct __declspec(dllexport) nested_exported {
    EXCLUDE_ATTR void fn_excluded();
    EXCLUDE_ATTR static int var_excluded;
  };
  // The same. nested_imported is not instantiated.
  struct __declspec(dllimport) nested_imported {
    EXCLUDE_ATTR void fn_excluded();
    EXCLUDE_ATTR static int var_excluded;
  };

  struct nested {
    EXCLUDE_ATTR __declspec(dllexport) void fn_excluded_exported(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
    EXCLUDE_ATTR __declspec(dllimport) void fn_excluded_imported(); // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
    EXCLUDE_ATTR __declspec(dllexport) static int var_excluded_exported; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
    EXCLUDE_ATTR __declspec(dllimport) static int var_excluded_imported; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  };
};

// Test that class-level dll attributes doesn't cause a warning on an excluded member.
template <class T>
struct __declspec(dllexport) DE {
  EXCLUDE_ATTR void fn_excluded();
};
template struct DE<int>;

template <class T>
struct __declspec(dllimport) DI {
  EXCLUDE_ATTR void fn_excluded();
};
template struct DI<int>;

// Test that dll attributes on explicit instantiation doesn't cause a warning on
// an excluded member.
// However, a non-template nested type may be warned on an excluded member by
// its dll attribute.
template <class T>
struct E {
  EXCLUDE_ATTR void fn_excluded();
  struct EXCLUDE_ATTR nested_excluded {
    __declspec(dllexport) void fn_exported();
    __declspec(dllimport) void fn_imported();
  };

  struct __declspec(dllexport) nested_exported_1 { // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
    EXCLUDE_ATTR void fn_excluded();
  };
  struct __declspec(dllimport) nested_imported_1 { // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
    EXCLUDE_ATTR void fn_excluded();
  };

  // Make sure that any warning isn't emitted if the nested type has no excluded members.
  struct __declspec(dllexport) nested_exported_2 {
    void fn();
  };
  struct __declspec(dllimport) nested_imported_2 {
    void fn();
  };
};
extern template struct __declspec(dllimport) E<long>;
template struct __declspec(dllexport) E<int>;
// expected-note@-1{{in instantiation of member class 'E<int>::nested_exported_1' requested here}}
// expected-note@-2{{in instantiation of member class 'E<int>::nested_imported_1' requested here}}
