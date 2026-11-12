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

// Additional test case from issue #59493
namespace init_list_array {

struct Base {
  int foox[5];
};

class Derived1 : public Base {
public:
  Derived1() : Base{{1,4,5,3,2}} {
    // The dereference to this->foox below should be initialized properly.
    clang_analyzer_dump(this->foox[0]); // expected-warning{{1 S32b}}
    clang_analyzer_dump(this->foox[1]); // expected-warning{{4 S32b}}
    clang_analyzer_dump(this->foox[2]); // expected-warning{{5 S32b}}
    clang_analyzer_dump(this->foox[3]); // expected-warning{{3 S32b}}
    clang_analyzer_dump(this->foox[4]); // expected-warning{{2 S32b}}
    clang_analyzer_dump(foox[0]); // expected-warning{{1 S32b}}
    clang_analyzer_dump(foox[1]); // expected-warning{{4 S32b}}
    clang_analyzer_dump(foox[2]); // expected-warning{{5 S32b}}
    clang_analyzer_dump(foox[3]); // expected-warning{{3 S32b}}
    clang_analyzer_dump(foox[4]); // expected-warning{{2 S32b}}
  }
};

void entry1() { Derived1 test; }

class Derived2 : public Base {
public:
  Derived2() : Base{{0}} {
    // The dereference to this->foox below should be initialized properly.
    clang_analyzer_dump(this->foox[0]); // expected-warning{{0 S32b}}
    clang_analyzer_dump(this->foox[1]); // expected-warning{{0 S32b}}
    clang_analyzer_dump(this->foox[2]); // expected-warning{{0 S32b}}
    clang_analyzer_dump(this->foox[3]); // expected-warning{{0 S32b}}
    clang_analyzer_dump(this->foox[4]); // expected-warning{{0 S32b}}
    clang_analyzer_dump(foox[0]); // expected-warning{{0 S32b}}
    clang_analyzer_dump(foox[1]); // expected-warning{{0 S32b}}
    clang_analyzer_dump(foox[2]); // expected-warning{{0 S32b}}
    clang_analyzer_dump(foox[3]); // expected-warning{{0 S32b}}
    clang_analyzer_dump(foox[4]); // expected-warning{{0 S32b}}
  }
};

void entry2() { Derived2 test; }

} // namespace init_list_array

namespace init_list_struct {

struct Base {
  struct { int a; int b; } foox;
};

class Derived : public Base {
public:
  Derived() : Base{{42, 24}} {
    // The dereference to this->foox below should be initialized properly.
    clang_analyzer_dump(this->foox.a); // expected-warning{{42 S32b}}
    clang_analyzer_dump(this->foox.b); // expected-warning{{24 S32b}}
    clang_analyzer_dump(foox.a); // expected-warning{{42 S32b}}
    clang_analyzer_dump(foox.b); // expected-warning{{24 S32b}}
  }
};

} // namespace init_list_struct

// Additional test case from issue 54533
namespace init_list_enum {

struct Base {
  enum : unsigned char { ZERO = 0, FORTY_TWO = 42 } foox;
};

class Derived : public Base {
public:
  Derived() : Base{FORTY_TWO} {
    // The dereference to this->foox below should be initialized properly.
    clang_analyzer_dump(this->foox); // expected-warning{{42 S32b}}
    clang_analyzer_dump(foox); // expected-warning{{42 S32b}}
  }
};

void entry() { Derived test; }

} // namespace init_list_enum
