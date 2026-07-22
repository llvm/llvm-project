// RUN: %check_clang_tidy %s cppcoreguidelines-special-member-functions %t -- \
// RUN:   -config="{CheckOptions: {cppcoreguidelines-special-member-functions.AllowImplicitlyDeletedCopyOrMove: true}}"

struct Base {
  Base(const Base &) = delete;
  Base &operator=(const Base &) = delete;
  Base(Base &&) = delete;
  Base &operator=(Base &&) = delete;
  ~Base();
};

struct A : Base {
  ~A() {}
};

struct Member {
  Base B;
  ~Member() {}
};

// CHECK-MESSAGES: [[@LINE+1]]:8: warning: class 'B' defines a non-default destructor and a move constructor but does not define a move assignment operator
struct B : Base {
  ~B() {}
  B(B &&);
};

// CHECK-MESSAGES: [[@LINE+1]]:8: warning: class 'C' defines a non-default destructor and a move assignment operator but does not define a move constructor
struct C : Base {
  ~C() {}
  C &operator=(C &&);
};
