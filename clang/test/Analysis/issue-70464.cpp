// RUN: %clang_analyze_cc1 %s -verify -analyzer-checker=core,debug.ExprInspection

// Refer to https://github.com/llvm/llvm-project/issues/70464 for more details.
//
// When the base class does not have a declared constructor, the base
// initializer in the constructor of the derived class should use the given
// initializer list to finish the initialization of the base class.
//
// Also testing base constructor and delegate constructor to make sure this
// change will not break these two cases when a CXXConstructExpr is available.

void clang_analyzer_dump(int);

namespace init_list {

struct Base {
  int foox;
};

class Derived : public Base {
public:
  Derived() : Base{42} {
    // The dereference to this->foox below should be initialized properly.
    clang_analyzer_dump(this->foox); // expected-warning{{42 S32b}}
    clang_analyzer_dump(foox); // expected-warning{{42 S32b}}
  }
};

void entry() { Derived test; }

} // namespace init_list

namespace base_ctor_call {

struct Base {
  int foox;
  Base(int x) : foox(x) {}
};

class Derived : public Base {
public:
  Derived() : Base{42} {
    clang_analyzer_dump(this->foox); // expected-warning{{42 S32b}}
    clang_analyzer_dump(foox); // expected-warning{{42 S32b}}
  }
};

void entry() { Derived test; }

} // namespace base_ctor_call

namespace delegate_ctor_call {

struct Base {
  int foox;
};

struct Derived : Base {
  Derived(int parx) { this->foox = parx; }
  Derived() : Derived{42} {
    clang_analyzer_dump(this->foox); // expected-warning{{42 S32b}}
    clang_analyzer_dump(foox); // expected-warning{{42 S32b}}
  }
};

void entry() { Derived test; }

} // namespace delegate_ctor_call
