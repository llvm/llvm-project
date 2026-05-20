// RUN: %check_clang_tidy %s cppcoreguidelines-use-default-member-init,modernize-use-default-member-init,cppcoreguidelines-explicit-constructor,misc-explicit-constructor %t

class Foo {
public:
  Foo() : _num(0)
  // CHECK-MESSAGES: warning: use default member initializer for '_num' [cppcoreguidelines-use-default-member-init,modernize-use-default-member-init]
  {}

  int use_the_members() const {
    return _num;
  }

private:
  int _num;
  // CHECK-FIXES: int _num{0};
};

class Bar {
public:
  Bar(int);
  // CHECK-MESSAGES: warning: single-argument constructors must be marked explicit to avoid unintentional implicit conversions [cppcoreguidelines-explicit-constructor,misc-explicit-constructor]
  // CHECK-FIXES: explicit Bar(int);
};
