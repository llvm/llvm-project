// RUN: %check_clang_tidy -std=c++23 %s misc-unconventional-assign-operator %t

struct BadArgument {
  BadArgument &operator=(this BadArgument& self, BadArgument &);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should take 'BadArgument const&', 'BadArgument&&' or 'BadArgument'
};

struct GoodArgument {
  GoodArgument &operator=(this GoodArgument& self, GoodArgument const &);
};
