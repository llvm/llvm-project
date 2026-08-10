// RUN: %clang_cc1 %s -triple i686-pc-win32 -std=c++11 -Wmicrosoft -fms-compatibility -verify
// RUN: not %clang_cc1 %s -triple i686-pc-win32 -std=c++11 -Wmicrosoft -fms-compatibility -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

struct X;
namespace name_at_tu_scope {
struct Y {
  friend struct X; // expected-warning-re {{unqualified friend declaration {{.*}} is a Microsoft extension}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:17-[[@LINE-1]]:17}:"::"
};
}

namespace enclosing_friend_decl {
struct B;
namespace ns {
struct A {
  friend struct B; // expected-warning-re {{unqualified friend declaration {{.*}} is a Microsoft extension}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:17-[[@LINE-1]]:17}:"enclosing_friend_decl::"
protected:
  A();
};
}
struct B {
  static void f() { ns::A x; }
};
}

namespace enclosing_friend_qualified {
struct B;
namespace ns {
struct A {
  friend struct enclosing_friend_qualified::B; // Adding name specifiers fixes it.
protected:
  A();
};
}
struct B {
  static void f() { ns::A x; }
};
}

namespace enclosing_friend_no_tag {
struct B;
namespace ns {
struct A {
  friend B; // Removing the tag decl fixes it.
protected:
  A();
};
}
struct B {
  static void f() { ns::A x; }
};
}

namespace enclosing_friend_func {
void f();
namespace ns {
struct A {
  // Amusingly, in MSVC, this declares ns::f(), and doesn't find the outer f().
  friend void f();
protected:
  A(); // expected-note {{declared protected here}}
};
}
void f() { ns::A x; } // expected-error {{calling a protected constructor of class 'enclosing_friend_func::ns::A'}}
}

namespace test_nns_fixit_hint {
namespace name1 {
namespace name2 {
struct X;
struct name2;
namespace name3 {
struct Y {
  friend struct X; // expected-warning-re {{unqualified friend declaration {{.*}} is a Microsoft extension}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:17-[[@LINE-1]]:17}:"name1::name2::"
};
}
}
}
}

// A friend declaration injects a forward declaration into the nearest enclosing
// non-member scope.
namespace friend_as_a_forward_decl {

class A {
  class Nested {
    friend class B;
    B *b;
  };
  B *b;
};
B *global_b;

void f() {
  class Local {
    friend class Z;
    Z *b;
  };
  Z *b;
}

}
