namespace std {
  template<typename T, unsigned size>
  struct array {
    T operator[](unsigned i) {
      return T{1};
    }
    T at(unsigned i) {
      return T{1};
    }
  };

  template<typename T>
  struct unique_ptr {
    T operator[](unsigned i) {
      return T{1};
    }
  };

  template<typename T>
  struct span {
    T operator[](unsigned i) {
      return T{1};
    }
  };
} // namespace std

namespace json {
  template<typename T>
  struct node{
    T operator[](unsigned i) {
      return T{1};
    }
  };
} // namespace json


// RUN: %check_clang_tidy %s cppcoreguidelines-prefer-at-over-subscript-operator %t
std::array<int, 3> a;

auto b = a[0];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]
auto c = a[1+1];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]
constexpr int index = 1;
auto d = a[index];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

int e(int index) {
  return a[index];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]
}

auto f = (&a)->operator[](1);
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

auto g = a.at(0);

std::unique_ptr<int> p;
auto q = p[0];

std::span<int> s;
auto t = s[0];

json::node<int> n;
auto m = n[0];
