// RUN: %check_clang_tidy -std=c++17-or-later %s readability-isolate-declaration %t

template <typename T1, typename T2>
struct pair {
  T1 first;
  T2 second;
  pair(T1 v1, T2 v2) : first(v1), second(v2) {}

  template <int N>
  decltype(auto) get() const {
    if constexpr (N == 0)
      return first;
    else if constexpr (N == 1)
      return second;
  }
};

void forbidden_transformations() {
  if (int i = 42, j = i; i == j)
    ;
  switch (int i = 12, j = 14; i)
    ;

  auto [i, j] = pair<int, int>(42, 42);
}

struct SomeClass {
  SomeClass() = default;
  SomeClass(int value);
};

namespace std {
template <typename T>
class initializer_list { const T *a, *b; };

template <typename T>
class vector {
public:
  vector() = default;
  vector(initializer_list<T> init) {}
};

class string {
public:
  string() = default;
  string(const char *) {}
};

namespace string_literals {
string operator""s(const char *, decltype(sizeof(int))) {
  return string();
}
} // namespace string_literals
} // namespace std

namespace Types {
typedef int MyType;
} // namespace Types

int touch1, touch2;

void modern() {
  auto autoInt1 = 3, autoInt2 = 4;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: auto autoInt1 = 3;
  // CHECK-FIXES: {{^  }}auto autoInt2 = 4;

  decltype(int()) declnottouch = 4;
  decltype(int()) declint1 = 5, declint2 = 3;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: decltype(int()) declint1 = 5;
  // CHECK-FIXES: {{^  }}decltype(int()) declint2 = 3;

  std::vector<int> vectorA = {1, 2}, vectorB = {1, 2, 3}, vectorC({1, 1, 1});
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: std::vector<int> vectorA = {1, 2};
  // CHECK-FIXES: {{^  }}std::vector<int> vectorB = {1, 2, 3};
  // CHECK-FIXES: {{^  }}std::vector<int> vectorC({1, 1, 1});

  using uType = int;
  uType utype1, utype2;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: uType utype1;
  // CHECK-FIXES: {{^  }}uType utype2;

  Types::MyType mytype1, mytype2, mytype3 = 3;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: Types::MyType mytype1;
  // CHECK-FIXES: {{^  }}Types::MyType mytype2;
  // CHECK-FIXES: {{^  }}Types::MyType mytype3 = 3;

  {
    using namespace std::string_literals;

    std::vector<std::string> s{"foo"s, "bar"s}, t{"foo"s}, u, a({"hey", "you"}), bb = {"h", "a"};
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: multiple declarations in a single statement reduces readability
    // CHECK-FIXES: std::vector<std::string> s{"foo"s, "bar"s};
    // CHECK-FIXES: {{^    }}std::vector<std::string> t{"foo"s};
    // CHECK-FIXES: {{^    }}std::vector<std::string> u;
    // CHECK-FIXES: {{^    }}std::vector<std::string> a({"hey", "you"});
    // CHECK-FIXES: {{^    }}std::vector<std::string> bb = {"h", "a"};
  }
}
