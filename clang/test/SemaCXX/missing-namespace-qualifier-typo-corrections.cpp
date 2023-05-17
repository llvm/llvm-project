// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c++11-extensions %s

namespace fizbin { class Foobar {}; } // expected-note 2 {{'fizbin::Foobar' declared here}} \
                                      // expected-note {{'Foobar' declared here}}
Foobar *my_bar  // expected-error{{unknown type name 'Foobar'; did you mean 'fizbin::Foobar'?}}
    = new Foobar; // expected-error{{unknown type name 'Foobar'; did you mean 'fizbin::Foobar'?}}
fizbin::Foobar *my_foo = new fizbin::FooBar; // expected-error{{no type named 'FooBar' in namespace 'fizbin'; did you mean 'Foobar'?}}

namespace barstool { int toFoobar() { return 1; } } // expected-note 3 {{'barstool::toFoobar' declared here}}
int Double(int x) { return x + x; }
void empty() {
  Double(toFoobar()); // expected-error{{use of undeclared identifier 'toFoobar'; did you mean 'barstool::toFoobar'?}}
}

namespace fizbin {
  namespace baztool { bool toFoobar() { return true; } } // expected-note{{'fizbin::baztool' declared here}}
  namespace nested { bool moreFoobar() { return true; } } // expected-note{{'fizbin::nested::moreFoobar' declared here}}
  namespace nested { bool lessFoobar() { return true; } } // expected-note{{'fizbin::nested' declared here}} \
                                                          // expected-note{{'fizbin::nested::lessFoobar' declared here}}
  class dummy { // expected-note 2 {{'fizbin::dummy' declared here}}
   public:
    static bool morebar() { return false; } // expected-note{{'morebar' declared here}}
  };
}
void Check() { // expected-note{{'Check' declared here}}
  if (toFoobar()) Double(7); // expected-error{{use of undeclared identifier 'toFoobar'; did you mean 'barstool::toFoobar'?}}
  if (noFoobar()) Double(7); // expected-error{{use of undeclared identifier 'noFoobar'; did you mean 'barstool::toFoobar'?}}
  if (moreFoobar()) Double(7); // expected-error{{use of undeclared identifier 'moreFoobar'; did you mean 'fizbin::nested::moreFoobar'}}
  if (lessFoobar()) Double(7); // expected-error{{use of undeclared identifier 'lessFoobar'; did you mean 'fizbin::nested::lessFoobar'?}}
  if (baztool::toFoobar()) Double(7); // expected-error{{use of undeclared identifier 'baztool'; did you mean 'fizbin::baztool'?}}
  if (nested::moreFoobar()) Double(7); // expected-error{{use of undeclared identifier 'nested'; did you mean 'fizbin::nested'?}}
  if (dummy::morebar()) Double(7); // expected-error{{use of undeclared identifier 'dummy'; did you mean 'fizbin::dummy'?}}
  if (dummy::mrebar()) Double(7); // expected-error{{use of undeclared identifier 'dummy'; did you mean 'fizbin::dummy'?}} \
                                     // expected-error{{no member named 'mrebar' in 'fizbin::dummy'; did you mean 'morebar'?}}
  if (moFoobin()) Double(7); // expected-error{{use of undeclared identifier 'moFoobin'}}
}

void Alt() {
  Cleck(); // expected-error{{use of undeclared identifier 'Cleck'; did you mean 'Check'?}}
}

namespace N {
  namespace inner {
    class myvector { /* ... */ }; // expected-note{{'inner::myvector' declared here}}
  }

  void f() {
    myvector v; // expected-error{{unknown type name 'myvector'; did you mean 'inner::myvector'?}}
  }
}

namespace realstd {
  inline namespace __1 {
    class mylinkedlist { /* ... */ }; // expected-note 2 {{'realstd::mylinkedlist' declared here}}
  }

  class linkedlist { /* ... */ };
}

void f() {
  mylinkedlist v; // expected-error{{unknown type name 'mylinkedlist'; did you mean 'realstd::mylinkedlist'?}}
  nylinkedlist w; // expected-error{{unknown type name 'nylinkedlist'; did you mean 'realstd::mylinkedlist'?}}
}

#if __cplusplus < 201703L
// Test case from http://llvm.org/bugs/show_bug.cgi?id=10318
namespace llvm {
 template <typename T> class GraphWriter {}; // expected-note 3{{declared here}}
}

struct S {};
void bar() {
 GraphWriter<S> x; //expected-error{{no template named 'GraphWriter'; did you mean 'llvm::GraphWriter'?}}
 (void)new llvm::GraphWriter; // expected-error {{use of class template 'llvm::GraphWriter' requires template arguments}}
 (void)new llvm::Graphwriter<S>; // expected-error {{no template named 'Graphwriter' in namespace 'llvm'; did you mean 'GraphWriter'?}}
}
#endif

// If namespace prefixes and character edits have the same weight, correcting
// "fimish" to "N::famish" would have the same edit distance as correcting
// "fimish" to "Finish". The result would be no correction being suggested
// unless one of the corrections is given precedence (e.g. by filtering out
// suggestions with added namespace qualifiers).
namespace N { void famish(int); }
void Finish(int); // expected-note {{'Finish' declared here}}
void Start() {
  fimish(7); // expected-error {{use of undeclared identifier 'fimish'; did you mean 'Finish'?}}
}

// But just eliminating the corrections containing added namespace qualifiers
// won't work if both of the tied corrections have namespace qualifiers added.
namespace N {
void someCheck(int); // expected-note {{'N::someCheck' declared here}}
namespace O { void somechock(int); }
}
void confusing() {
  somechick(7); // expected-error {{use of undeclared identifier 'somechick'; did you mean 'N::someCheck'?}}
}


class Message {};
namespace extra {
  namespace util {
    namespace MessageUtils {
      bool Equivalent(const Message&, const Message&); // expected-note {{'extra::util::MessageUtils::Equivalent' declared here}} \
                                                       // expected-note {{'::extra::util::MessageUtils::Equivalent' declared here}}
    }
  }
}
namespace util { namespace MessageUtils {} }
bool nstest () {
  Message a, b;
  return util::MessageUtils::Equivalent(a, b); // expected-error {{no member named 'Equivalent' in namespace 'util::MessageUtils'; did you mean 'extra::util::MessageUtils::Equivalent'?}}
}

namespace util {
  namespace extra {
    bool nstest () {
      Message a, b;
      return MessageUtils::Equivalent(a, b); // expected-error {{no member named 'Equivalent' in namespace 'util::MessageUtils'; did you mean '::extra::util::MessageUtils::Equivalent'?}}
    }
  }
}
