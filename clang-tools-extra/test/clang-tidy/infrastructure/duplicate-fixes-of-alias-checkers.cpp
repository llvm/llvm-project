// RUN: %check_clang_tidy %s cppcoreguidelines-use-default-member-init,modernize-use-default-member-init %t

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
