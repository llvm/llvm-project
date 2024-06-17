// RUN: %check_clang_tidy %s \
// RUN: cppcoreguidelines-prefer-at-over-subscript-operator %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-prefer-at-over-subscript-operator.ExcludeClasses: "ExcludedClass1;ExcludedClass2"}}'

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

  template<typename T, typename V>
  struct map {
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

struct SubClass : std::array<int, 3> {};

class ExcludedClass1 {
  public:
    int operator[](unsigned i) {
      return 1;
    }
    int at(unsigned i) {
      return 1;
    }
};

class ExcludedClass2 {
  public:
    int operator[](unsigned i) {
      return 1;
    }
    int at(unsigned i) {
      return 1;
    }
};

std::array<int, 3> a;

auto b = a[0];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

auto c = a[1+1];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

constexpr int Index = 1;
auto d = a[Index];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

int e(int Ind) {
  return a[Ind];
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

SubClass Sub;
auto r = Sub[0];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

typedef std::array<int, 3> ar;
ar BehindDef;
auto u = BehindDef[0];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

template<typename T> int TestTemplate(T t){
  return t[0];
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

}

auto v = TestTemplate<>(a);
auto w = TestTemplate<>(p);

//explicitely excluded classes / struct / template
ExcludedClass1 E1;
auto x1 = E1[0];

ExcludedClass2 E2;
auto x2 = E1[0];

std::map<int,int> TestMap;
auto y = TestMap[0];

#define SUBSCRIPT_BEHIND_MARCO(x) a[x]
#define ARG_BEHIND_MACRO 0
#define OBJECT_BEHIND_MACRO a

auto m1 = SUBSCRIPT_BEHIND_MARCO(0);
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

auto m2 = a[ARG_BEHIND_MACRO];
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]

auto m3 = OBJECT_BEHIND_MACRO[0];
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: found possibly unsafe operator[], consider using at() instead [cppcoreguidelines-prefer-at-over-subscript-operator]
