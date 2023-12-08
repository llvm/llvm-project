// RUN: %check_clang_tidy -std=c++98,c++03 %s misc-unconventional-assign-operator %t

struct BadArgument {
  BadArgument &operator=(BadArgument &);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should take 'BadArgument const&' or 'BadArgument'
};
