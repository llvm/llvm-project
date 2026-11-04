// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-keyword %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ -std=c++2c %s
// good-no-diagnostics

// 'try', 'this' and 'throw' are not tested as identifiers, but are instead
// tested as other constructs (otherwise there would be redefinition errors in
// C).
int catch;              // expected-warning {{identifier 'catch' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int class;              // expected-warning {{identifier 'class' conflicts with a C++ keyword}} \
                           cxx-error {{declaration of anonymous class must be a definition}} \
                           cxx-warning {{declaration does not declare anything}}
int const_cast;         // expected-warning {{identifier 'const_cast' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int delete;             // expected-warning {{identifier 'delete' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int dynamic_cast;       // expected-warning {{identifier 'dynamic_cast' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int explicit;           // expected-warning {{identifier 'explicit' conflicts with a C++ keyword}} \
                           cxx-error {{'explicit' can only appear on non-static member functions}} \
                           cxx-warning {{declaration does not declare anything}}
int export;             // expected-warning {{identifier 'export' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int friend;             // expected-warning {{identifier 'friend' conflicts with a C++ keyword}} \
                           cxx-error {{'friend' used outside of class}} \
                           cxx-warning {{declaration does not declare anything}}
int mutable;            // expected-warning {{identifier 'mutable' conflicts with a C++ keyword}} \
                           cxx-warning {{declaration does not declare anything}}
int namespace;          // expected-warning {{identifier 'namespace' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int new;                // expected-warning {{identifier 'new' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int operator;           // expected-warning {{identifier 'operator' conflicts with a C++ keyword}} \
                           cxx-error {{expected a type}}
int private;            // expected-warning {{identifier 'private' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int protected;          // expected-warning {{identifier 'protected' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int public;             // expected-warning {{identifier 'public' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int reinterpret_cast;   // expected-warning {{identifier 'reinterpret_cast' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int static_cast;        // expected-warning {{identifier 'static_cast' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int template;           // expected-warning {{identifier 'template' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int typename;           // expected-warning {{identifier 'typename' conflicts with a C++ keyword}} \
                           cxx-error {{expected a qualified name after 'typename'}}
int typeid;             // expected-warning {{identifier 'typeid' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int using;              // expected-warning {{identifier 'using' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int virtual;            // expected-warning {{identifier 'virtual' conflicts with a C++ keyword}} \
                           cxx-error {{'virtual' can only appear on non-static member functions}} \
                           cxx-warning {{declaration does not declare anything}}
int wchar_t;            // expected-warning {{identifier 'wchar_t' conflicts with a C++ keyword}} \
                           cxx-error {{cannot combine with previous 'int' declaration specifier}} \
                           cxx-warning {{declaration does not declare anything}}
int char8_t;            // expected-warning {{identifier 'char8_t' conflicts with a C++ keyword}} \
                           cxx-error {{cannot combine with previous 'int' declaration specifier}} \
                           cxx-warning {{declaration does not declare anything}}
int char16_t;           // expected-warning {{identifier 'char16_t' conflicts with a C++ keyword}} \
                           cxx-error {{cannot combine with previous 'int' declaration specifier}} \
                           cxx-warning {{declaration does not declare anything}}
int char32_t;           // expected-warning {{identifier 'char32_t' conflicts with a C++ keyword}} \
                           cxx-error {{cannot combine with previous 'int' declaration specifier}} \
                           cxx-warning {{declaration does not declare anything}}
int noexcept;           // expected-warning {{identifier 'noexcept' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int co_await;           // expected-warning {{identifier 'co_await' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int co_return;          // expected-warning {{identifier 'co_return' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int co_yield;           // expected-warning {{identifier 'co_yield' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int consteval;          // expected-warning {{identifier 'consteval' conflicts with a C++ keyword}} \
                           cxx-error {{consteval can only be used in function declarations}}
int constinit;          // expected-warning {{identifier 'constinit' conflicts with a C++ keyword}} \
                           cxx-error {{constinit can only be used in variable declarations}}
int concept;            // expected-warning {{identifier 'concept' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}
int requires;           // expected-warning {{identifier 'requires' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}

// Now try the same thing, but as struct members.
struct S {
  int catch;            // expected-warning {{identifier 'catch' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int class;            // expected-warning {{identifier 'class' conflicts with a C++ keyword}} \
                           cxx-error {{declaration of anonymous class must be a definition}} \
                           cxx-warning {{declaration does not declare anything}}
  int const_cast;       // expected-warning {{identifier 'const_cast' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int delete;           // expected-warning {{identifier 'delete' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int dynamic_cast;     // expected-warning {{identifier 'dynamic_cast' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int explicit;         // expected-warning {{identifier 'explicit' conflicts with a C++ keyword}} \
                           cxx-error {{'explicit' can only appear on non-static member functions}} \
                           cxx-warning {{declaration does not declare anything}}
  int export;           // expected-warning {{identifier 'export' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int friend;           // expected-warning {{identifier 'friend' conflicts with a C++ keyword}} \
                           cxx-error {{'friend' must appear first in a non-function declaration}}
  int mutable;          // expected-warning {{identifier 'mutable' conflicts with a C++ keyword}} \
                           cxx-warning {{declaration does not declare anything}}
  int namespace;        // expected-warning {{identifier 'namespace' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int new;              // expected-warning {{identifier 'new' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int operator;         // expected-warning {{identifier 'operator' conflicts with a C++ keyword}} \
                           cxx-error {{expected a type}}
  int private;          // expected-warning {{identifier 'private' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int protected;        // expected-warning {{identifier 'protected' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int public;           // expected-warning {{identifier 'public' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int reinterpret_cast; // expected-warning {{identifier 'reinterpret_cast' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int static_cast;      // expected-warning {{identifier 'static_cast' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int template;         // expected-warning {{identifier 'template' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int this;             // expected-warning {{identifier 'this' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int throw;            // expected-warning {{identifier 'throw' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int try;              // expected-warning {{identifier 'try' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int typename;         // expected-warning {{identifier 'typename' conflicts with a C++ keyword}} \
                           cxx-error {{expected a qualified name after 'typename'}}
  int typeid;           // expected-warning {{identifier 'typeid' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int using;            // expected-warning {{identifier 'using' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int virtual;          // expected-warning {{identifier 'virtual' conflicts with a C++ keyword}} \
                           cxx-error {{'virtual' can only appear on non-static member functions}} \
                           cxx-warning {{declaration does not declare anything}}
  int wchar_t;          // expected-warning {{identifier 'wchar_t' conflicts with a C++ keyword}} \
                           cxx-error {{cannot combine with previous 'int' declaration specifier}} \
                           cxx-warning {{declaration does not declare anything}}
  int char8_t;          // expected-warning {{identifier 'char8_t' conflicts with a C++ keyword}} \
                           cxx-error {{cannot combine with previous 'int' declaration specifier}} \
                           cxx-warning {{declaration does not declare anything}}
  int char16_t;         // expected-warning {{identifier 'char16_t' conflicts with a C++ keyword}} \
                           cxx-error {{cannot combine with previous 'int' declaration specifier}} \
                           cxx-warning {{declaration does not declare anything}}
  int char32_t;         // expected-warning {{identifier 'char32_t' conflicts with a C++ keyword}} \
                           cxx-error {{cannot combine with previous 'int' declaration specifier}} \
                           cxx-warning {{declaration does not declare anything}}
  int noexcept;         // expected-warning {{identifier 'noexcept' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int co_await;         // expected-warning {{identifier 'co_await' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int co_return;        // expected-warning {{identifier 'co_return' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int co_yield;         // expected-warning {{identifier 'co_yield' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}}
  int consteval;        // expected-warning {{identifier 'consteval' conflicts with a C++ keyword}} \
                           cxx-error {{consteval can only be used in function declarations}}
  int constinit;        // expected-warning {{identifier 'constinit' conflicts with a C++ keyword}} \
                           cxx-error {{constinit can only be used in variable declarations}}
  int concept;          // expected-warning {{identifier 'concept' conflicts with a C++ keyword}} \
                           cxx-error {{concept declarations may only appear in global or namespace scope}}
  int requires;         // expected-warning {{identifier 'requires' conflicts with a C++ keyword}} \
                           cxx-error {{expected member name or ';' after declaration specifiers}} \
                           cxx-error {{trailing requires clause can only be used when declaring a function}} \
                           cxx-error {{expected expression}}
};

// Smoke test that we catch a keyword used as an enumerator. If we diagnose
// one, we'll diagnose them all.
enum E {
  throw, // expected-warning {{identifier 'throw' conflicts with a C++ keyword}} \
            cxx-error {{expected identifier}}
};

// Smoke test that we catch a keyword used as a tag name.
struct try { // expected-warning {{identifier 'try' conflicts with a C++ keyword}} \
                cxx-error {{declaration of anonymous struct must be a definition}} \
                cxx-warning {{declaration does not declare anything}}
  int x;
};

// Smoke test that we catch keyword use in a function name.
void this(void);        // expected-warning {{identifier 'this' conflicts with a C++ keyword}} \
                           cxx-error {{expected unqualified-id}}

// Smoke test that we catch keyword use in function parameters too.
void func(int private); // expected-warning {{identifier 'private' conflicts with a C++ keyword}} \
                           cxx-error {{invalid parameter name: 'private' is a keyword}}

// These are conditionally a keyword in C++, so they're intentionally not being
// diagnosed as a keyword.
int module;
int import;
int override;
int final;

// We do not diagnose use of C++ keywords when used as a macro name because
// that does not conflict with C++ (the macros will be replaced before the
// keyword is seen by the parser).
#define this 12

// FIXME: These tests are disabled for C++ because it causes a crash.
// See GH114815.
#ifndef __cplusplus
int decltype;           // expected-warning {{identifier 'decltype' conflicts with a C++ keyword}}
struct T {
  int decltype;         // expected-warning {{identifier 'decltype' conflicts with a C++ keyword}}
};
#endif // __cplusplus

// Check alternative operator names.
int and;      // expected-warning {{identifier 'and' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
int and_eq;   // expected-warning {{identifier 'and_eq' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
int bitand;   // expected-warning {{identifier 'bitand' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
int bitor;    // expected-warning {{identifier 'bitor' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
int compl;    // expected-warning {{identifier 'compl' conflicts with a C++ keyword}} \
                 cxx-error {{expected a class name after '~' to name a destructor}}
int not;      // expected-warning {{identifier 'not' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
int not_eq;   // expected-warning {{identifier 'not_eq' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
int or;       // expected-warning {{identifier 'or' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
int or_eq;    // expected-warning {{identifier 'or_eq' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
int xor;      // expected-warning {{identifier 'xor' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
int xor_eq;   // expected-warning {{identifier 'xor_eq' conflicts with a C++ keyword}} \
                 cxx-error {{expected unqualified-id}}
