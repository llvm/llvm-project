// RUN: %check_clang_tidy %s google-runtime-member-string-references %t

namespace std {
template<typename T>
  class basic_string {};

typedef basic_string<char> string;
}

class string {};


struct A {
  const std::string &s;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: const string& members are dangerous; it is much better to use alternatives, such as pointers or simple constants [google-runtime-member-string-references]
};

struct B {
  std::string &s;
};

struct C {
  const std::string s;
};

template <typename T>
struct D {
  D();
  const T &s;
  const std::string &s2;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: const string& members are dangerous
};

D<std::string> d;

struct AA {
  const string &s;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: const string& members are dangerous
};

struct BB {
  string &s;
};

struct CC {
  const string s;
};

D<string> dd;
