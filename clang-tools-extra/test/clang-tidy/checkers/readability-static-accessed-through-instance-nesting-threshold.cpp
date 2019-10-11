// RUN: %check_clang_tidy %s readability-static-accessed-through-instance %t -- -config="{CheckOptions: [{key: readability-static-accessed-through-instance.NameSpecifierNestingThreshold, value: 4}]}" --

// Nested specifiers
namespace M {
namespace N {
struct V {
  static int v;
  struct T {
    static int t;
    struct U {
      static int u;
    };
  };
};
}
}

void f(M::N::V::T::U u) {
  M::N::V v;
  v.v = 12;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  M::N::V::v = 12;{{$}}

  M::N::V::T w;
  w.t = 12;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  M::N::V::T::t = 12;{{$}}

  // u.u is not changed, because the nesting level is over 4
  u.u = 12;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: static member
  // CHECK-FIXES: {{^}}  u.u = 12;{{$}}
}
