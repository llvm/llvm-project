// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -fsyntax-only -verify=expected,msc %s
// RUN: %clang_cc1 -triple x86_64-mingw                 -fsyntax-only -verify=expected,gnu %s
// RUN: %clang_cc1 -triple x86_64-cygwin                -fsyntax-only -verify=expected,gnu %s

// Test that attaching the exclude_from_explicit_instantiation attribute and
// either the dllexport or dllimport attribute together causes a warning.
// One of them is ignored, depending on the context that is declared.

#define EXCLUDE_ATTR __attribute__((exclude_from_explicit_instantiation))

// Test that exclude_from_explicit_instantiation takes precedence over
// dllexport/dllimport in a template context.
template <class T>
struct class_tmpl_no_instantiated {
  EXCLUDE_ATTR __declspec(dllexport) void fn_excluded_exported(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  EXCLUDE_ATTR __declspec(dllimport) void fn_excluded_imported(); // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  __declspec(dllexport) EXCLUDE_ATTR void fn_exported_excluded(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  __declspec(dllimport) EXCLUDE_ATTR void fn_imported_excluded(); // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

  EXCLUDE_ATTR __declspec(dllexport) static int var_excluded_exported; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  EXCLUDE_ATTR __declspec(dllimport) static int var_excluded_imported; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  __declspec(dllexport) EXCLUDE_ATTR static int var_exported_excluded; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  __declspec(dllimport) EXCLUDE_ATTR static int var_imported_excluded; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

  struct EXCLUDE_ATTR __declspec(dllexport) nested_excluded_exported {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  struct EXCLUDE_ATTR __declspec(dllimport) nested_excluded_imported {}; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  struct __declspec(dllexport) EXCLUDE_ATTR nested_exported_excluded {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  struct __declspec(dllimport) EXCLUDE_ATTR nested_imported_excluded {}; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

  template <class U>
  struct EXCLUDE_ATTR __declspec(dllexport) nested_tmpl_excluded_exported {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  template <class U>
  EXCLUDE_ATTR __declspec(dllexport) static T var_template_excluded; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  template <class U>
  EXCLUDE_ATTR __declspec(dllexport) void fn_template_excluded(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

  struct nested {
    EXCLUDE_ATTR __declspec(dllexport) void fn_excluded_exported(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  };

  struct EXCLUDE_ATTR nested_excluded {
    __declspec(dllexport) void fn_exported();
  };
  struct __declspec(dllexport) nested_exported {
    EXCLUDE_ATTR void fn_excluded();
  };
  struct __declspec(dllimport) nested_imported {
    EXCLUDE_ATTR void fn_excluded();
  };

  template <class U>
  struct nested_tmpl {
    EXCLUDE_ATTR __declspec(dllexport) void fn_excluded_exported(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  };
};

// Test that a class-level dll attribute doesn't cause a warning on an excluded member.
template <class T>
struct __declspec(dllexport) class_tmpl_exported {
  EXCLUDE_ATTR void fn_excluded();
};
template struct class_tmpl_exported<int>;
void use_class_tmpl_exported() { class_tmpl_exported<long>().fn_excluded(); }

template <class T>
struct __declspec(dllimport) class_tmpl_imported {
  EXCLUDE_ATTR void fn_excluded();
};
template struct class_tmpl_imported<int>;
void use_class_tmpl_imported() { class_tmpl_imported<long>().fn_excluded(); }

// Test that a dll attribute on an explicit instantiation doesn't cause a warning on
// an excluded member.
template <class T>
struct class_tmpl_explicit_inst {
  EXCLUDE_ATTR void fn_excluded();
  EXCLUDE_ATTR static T var_excluded;
  struct EXCLUDE_ATTR nested_excluded {
    __declspec(dllexport) void fn_exported();
    __declspec(dllimport) void fn_imported();
  };

  struct __declspec(dllexport) nested_exported {
    EXCLUDE_ATTR void fn_excluded();
  };
  struct __declspec(dllimport) nested_imported {
    EXCLUDE_ATTR void fn_excluded();
  };
};
extern template struct __declspec(dllimport) class_tmpl_explicit_inst<long>;
template struct __declspec(dllexport) class_tmpl_explicit_inst<int>;

// Test that exclude_from_explicit_instantiation is ignored in a non-template context.
struct class_nontmpl {
  EXCLUDE_ATTR __declspec(dllexport) void fn_excluded_exported(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  EXCLUDE_ATTR __declspec(dllimport) void fn_excluded_imported(); // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  __declspec(dllexport) EXCLUDE_ATTR void fn_exported_excluded(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  __declspec(dllimport) EXCLUDE_ATTR void fn_imported_excluded(); // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

  EXCLUDE_ATTR __declspec(dllexport) static int var_excluded_exported; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  EXCLUDE_ATTR __declspec(dllimport) static int var_excluded_imported; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  __declspec(dllexport) EXCLUDE_ATTR static int var_exported_excluded; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  __declspec(dllimport) EXCLUDE_ATTR static int var_imported_excluded; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

  struct EXCLUDE_ATTR __declspec(dllexport) nested_excluded_exported {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  struct EXCLUDE_ATTR __declspec(dllimport) nested_excluded_imported {}; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  struct __declspec(dllexport) EXCLUDE_ATTR nested_exported_excluded {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  struct __declspec(dllimport) EXCLUDE_ATTR nested_imported_excluded {}; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

  template <class T>
  struct EXCLUDE_ATTR __declspec(dllexport) class_template_excluded {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  template <class T>
  EXCLUDE_ATTR __declspec(dllexport) static T var_template_excluded; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  template <class T>
  EXCLUDE_ATTR __declspec(dllexport) void fn_template_excluded(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

  struct nested {
    EXCLUDE_ATTR __declspec(dllexport) void fn_excluded_exported(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  };

  struct EXCLUDE_ATTR nested_excluded {
    __declspec(dllexport) void fn_excluded_exported();
  };
  struct __declspec(dllexport) nested_exported {
    EXCLUDE_ATTR void fn_excluded();
  };
  struct __declspec(dllimport) nested_imported {
    EXCLUDE_ATTR void fn_excluded();
  };

  template <class T>
  struct nested_tmpl {
    EXCLUDE_ATTR __declspec(dllexport) void fn_excluded_exported(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
  };
};

// Test that exclude_from_explicit_instantiation is ignored on a non-member entity.
EXCLUDE_ATTR __declspec(dllexport) void fn_excluded_exported(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
EXCLUDE_ATTR __declspec(dllimport) void fn_excluded_imported(); // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
__declspec(dllexport) EXCLUDE_ATTR void fn_exported_excluded(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
__declspec(dllimport) EXCLUDE_ATTR void fn_imported_excluded(); // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

EXCLUDE_ATTR __declspec(dllexport) int var_excluded_exported; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
EXCLUDE_ATTR __declspec(dllimport) int var_excluded_imported; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
__declspec(dllexport) EXCLUDE_ATTR int var_exported_excluded; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
__declspec(dllimport) EXCLUDE_ATTR int var_imported_excluded; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

struct EXCLUDE_ATTR __declspec(dllexport) class_excluded_exported {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
struct EXCLUDE_ATTR __declspec(dllimport) class_excluded_imported {}; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
struct __declspec(dllexport) EXCLUDE_ATTR class_exported_excluded {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
struct __declspec(dllimport) EXCLUDE_ATTR class_imported_excluded {}; // expected-warning{{'dllimport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

template <class T>
struct EXCLUDE_ATTR __declspec(dllexport) class_tmpl_excluded_exported {}; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
template <class T>
EXCLUDE_ATTR __declspec(dllexport) T var_template_excluded; // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}
template <class T>
EXCLUDE_ATTR __declspec(dllexport) void fn_template_excluded(); // expected-warning{{'dllexport' attribute ignored; 'exclude_from_explicit_instantiation' takes precedence}}

EXCLUDE_ATTR void fn_excluded();

EXCLUDE_ATTR int var_excluded;

struct EXCLUDE_ATTR class_excluded {
  __declspec(dllexport) void fn_excluded_exported();
};
struct __declspec(dllexport) class_exported {
  EXCLUDE_ATTR void fn_excluded();
};

// Test that an excluded member in an imported class can have its definition without any warning.
struct __declspec(dllimport) class_imported { // #import
  void fn(); // expected-note{{previous declaration is here}}
  EXCLUDE_ATTR void fn_excluded();
  static int var;
  EXCLUDE_ATTR static int var_excluded;
};

void class_imported::fn() {}
  // msc-warning@-1{{'class_imported::fn' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
  // gnu-warning@-2{{'class_imported::fn' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
  // gnu-note@#import {{previous attribute is here}}

void class_imported::fn_excluded() {}

int class_imported::var = 0;
  // expected-error@-1{{definition of dllimport static field not allowed}}
  // expected-note@#import{{attribute is here}}

int class_imported::var_excluded = 0;
