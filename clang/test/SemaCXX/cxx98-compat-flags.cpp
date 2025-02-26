// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++98-compat-pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++98-compat-pedantic -Wno-bind-to-temporary-copy -Wno-unnamed-type-template-args -Wno-local-type-template-args -Wno-binary-literal -Werror %s

template<typename T> int TemplateFn(T) { return 0; }
void LocalTemplateArg() {
  struct S {};
  TemplateFn(S()); // expected-warning {{local type 'S' as template argument is incompatible with C++98}}
                   // expected-note@-1 {{while substituting deduced template arguments}}
}
struct {} obj_of_unnamed_type; // expected-note {{here}}
int UnnamedTemplateArg = TemplateFn(obj_of_unnamed_type); // expected-warning {{unnamed type as template argument is incompatible with C++98}}
                                                          // expected-note@-1 {{while substituting deduced template arguments}}

namespace CopyCtorIssues {
  struct Private {
    Private();
  private:
    Private(const Private&); // expected-note {{declared private here}}
  };
  struct NoViable {
    NoViable(); // expected-note {{not viable}}
    NoViable(NoViable&); // expected-note {{not viable}}
  };
  struct Ambiguous {
    Ambiguous();
    Ambiguous(const Ambiguous &, int = 0); // expected-note {{candidate}}
    Ambiguous(const Ambiguous &, double = 0); // expected-note {{candidate}}
  };
  struct Deleted {
    Private p; // expected-note {{copy constructor of 'Deleted' is implicitly deleted because field 'p' has an inaccessible copy constructor}}
  };

  const Private &a = Private(); // expected-warning {{copying variable of type 'Private' when binding a reference to a temporary would invoke an inaccessible constructor in C++98}}
  const NoViable &b = NoViable(); // expected-warning {{copying variable of type 'NoViable' when binding a reference to a temporary would find no viable constructor in C++98}}
  const Ambiguous &c = Ambiguous(); // expected-warning {{copying variable of type 'Ambiguous' when binding a reference to a temporary would find ambiguous constructors in C++98}}
  const Deleted &d = Deleted(); // expected-warning {{copying variable of type 'Deleted' when binding a reference to a temporary would invoke a deleted constructor in C++98}}

  int n = 0b00100101001; // expected-warning {{binary integer literals are incompatible with C++ standards before C++14}}
}
