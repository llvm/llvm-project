// RUN: %check_clang_tidy %s cppcoreguidelines-use-default-member-init,modernize-use-default-member-init %t -- \
//// RUN:     -config='{CheckOptions: { \
//// RUN:         cppcoreguidelines-use-default-member-init.UseAssignment: true, \
//// RUN:     }}'

class Foo {
public:
  Foo() : _num(0)
  // CHECK-MESSAGES: warning: use default member initializer for '_num' [cppcoreguidelines-use-default-member-init,modernize-use-default-member-init]
  // CHECK-MESSAGES: note: cannot apply fix-it because an alias checker has suggested a different fix-it; please remove one of the checkers ('cppcoreguidelines-use-default-member-init', 'modernize-use-default-member-init') or ensure they are both configured the same
  {}

  int use_the_members() const {
    return _num;
  }

private:
  int _num;
  // CHECK-FIXES: int _num;
};
