// RUN: %check_clang_tidy --match-partial-fixes -std=c++14-or-later %s readability-container-size-empty %t -- \
// RUN: -config="{CheckOptions: {readability-container-size-empty.ExcludedComparisonTypes: '::std::array;::IgnoredDummyType'}}" \
// RUN: -- -fno-delayed-template-parsing -isystem %clang_tidy_headers
#include <string>

namespace std {
template <typename T> struct vector {
  vector();
  bool operator==(const vector<T>& other) const;
  bool operator!=(const vector<T>& other) const;
  unsigned long size() const;
  bool empty() const;
};

inline namespace __v2 {
template <typename T> struct set {
  set();
  bool operator==(const set<T>& other) const;
  bool operator!=(const set<T>& other) const;
  unsigned long size() const;
  bool empty() const;
};
}

namespace string_literals{
string operator""s(const char *, size_t);
}

}

template <typename T>
class TemplatedContainer {
public:
  bool operator==(const TemplatedContainer<T>& other) const;
  bool operator!=(const TemplatedContainer<T>& other) const;
  unsigned long size() const;
  bool empty() const;
};

template <typename T>
class PrivateEmpty {
public:
  bool operator==(const PrivateEmpty<T>& other) const;
  bool operator!=(const PrivateEmpty<T>& other) const;
  unsigned long size() const;
private:
  bool empty() const;
};

struct BoolSize {
  bool size() const;
  bool empty() const;
};

struct EnumSize {
  enum E { one };
  enum E size() const;
  bool empty() const;
};

class Container {
public:
  bool operator==(const Container& other) const;
  unsigned long size() const;
  bool empty() const;
};

class Derived : public Container {
};

class Container2 {
public:
  unsigned long size() const;
  bool empty() const { return size() == 0; }
};

class Container3 {
public:
  unsigned long size() const;
  bool empty() const;
};

bool Container3::empty() const { return this->size() == 0; }

class Container4 {
public:
  bool operator==(const Container4& rhs) const;
  unsigned long size() const;
  bool empty() const { return *this == Container4(); }
};

struct Lazy {
  constexpr unsigned size() const { return 0; }
  constexpr bool empty() const { return true; }
};

std::string s_func() {
  return std::string();
}

void takesBool(bool)
{

}

bool returnsBool() {
  std::set<int> intSet;
  std::string str;
  std::string str2;
  std::wstring wstr;
  (void)(str.size() + 0);
  (void)(str.length() + 0);
  (void)(str.size() - 0);
  (void)(str.length() - 0);
  (void)(0 + str.size());
  (void)(0 + str.length());
  (void)(0 - str.size());
  (void)(0 - str.length());
  if (intSet.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-FIXES: {{^  }}if (intSet.empty()){{$}}
  // CHECK-MESSAGES: :21:8: note: method 'set'::empty() defined here
  if (intSet == std::set<int>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness
  // CHECK-FIXES: {{^  }}if (intSet.empty()){{$}}
  // CHECK-MESSAGES: :21:8: note: method 'set'::empty() defined here
  if (s_func() == "")
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (s_func().empty()){{$}}
  if (str.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size'
  // CHECK-FIXES: {{^  }}if (str.empty()){{$}}
  if (str.length() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'length'
  // CHECK-FIXES: {{^  }}if (str.empty()){{$}}
  if ((str + str2).size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size'
  // CHECK-FIXES: {{^  }}if ((str + str2).empty()){{$}}
  if ((str + str2).length() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'length'
  // CHECK-FIXES: {{^  }}if ((str + str2).empty()){{$}}
  if (str == "")
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (str.empty()){{$}}
  if (str + str2 == "")
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if ((str + str2).empty()){{$}}
  if (wstr.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size'
  // CHECK-FIXES: {{^  }}if (wstr.empty()){{$}}
  if (wstr.length() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'length'
  // CHECK-FIXES: {{^  }}if (wstr.empty()){{$}}
  if (wstr == L"")
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (wstr.empty()){{$}}
  std::vector<int> vect;
  if (vect.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size'
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (vect == std::vector<int>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (vect.size() != 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (vect != std::vector<int>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (0 == vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (0 != vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (std::vector<int>() == vect)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (std::vector<int>() != vect)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (vect.size() > 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (0 < vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (vect.size() < 1)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (1 > vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (vect.size() >= 1)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (1 <= vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (vect.size() > 1) // no warning
    ;
  if (1 < vect.size()) // no warning
    ;
  if (vect.size() <= 1) // no warning
    ;
  if (1 >= vect.size()) // no warning
    ;
  if (!vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}

  if (vect.empty())
    ;

  const std::vector<int> vect2;
  if (vect2.size() != 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect2.empty()){{$}}

  std::vector<int> *vect3 = new std::vector<int>();
  if (vect3->size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect3->empty()){{$}}
  if ((*vect3).size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if ((*vect3).empty()){{$}}
  if ((*vect3) == std::vector<int>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect3->empty()){{$}}
  if (*vect3 == std::vector<int>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect3->empty()){{$}}

  delete vect3;

  const std::vector<int> &vect4 = vect2;
  if (vect4.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect4.empty()){{$}}
  if (vect4 == std::vector<int>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect4.empty()){{$}}

  TemplatedContainer<void> templated_container;
  if (templated_container.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (templated_container.empty()){{$}}
  if (templated_container == TemplatedContainer<void>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (templated_container.empty()){{$}}
  if (templated_container.size() != 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (templated_container != TemplatedContainer<void>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (0 == templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (templated_container.empty()){{$}}
  if (TemplatedContainer<void>() == templated_container)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (templated_container.empty()){{$}}
  if (0 != templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (TemplatedContainer<void>() != templated_container)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (templated_container.size() > 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (0 < templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (templated_container.size() < 1)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (templated_container.empty()){{$}}
  if (1 > templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (templated_container.empty()){{$}}
  if (templated_container.size() >= 1)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (1 <= templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (templated_container.size() > 1) // no warning
    ;
  if (1 < templated_container.size()) // no warning
    ;
  if (templated_container.size() <= 1) // no warning
    ;
  if (1 >= templated_container.size()) // no warning
    ;
  if (!templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (templated_container.empty()){{$}}
  if (templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}

  if (templated_container.empty())
    ;

  // No warnings expected.
  PrivateEmpty<void> private_empty;
  if (private_empty.size() == 0)
    ;
  if (private_empty == PrivateEmpty<void>())
    ;
  if (private_empty.size() != 0)
    ;
  if (private_empty != PrivateEmpty<void>())
    ;
  if (0 == private_empty.size())
    ;
  if (PrivateEmpty<void>() == private_empty)
    ;
  if (0 != private_empty.size())
    ;
  if (PrivateEmpty<void>() != private_empty)
    ;
  if (private_empty.size() > 0)
    ;
  if (0 < private_empty.size())
    ;
  if (private_empty.size() < 1)
    ;
  if (1 > private_empty.size())
    ;
  if (private_empty.size() >= 1)
    ;
  if (1 <= private_empty.size())
    ;
  if (private_empty.size() > 1)
    ;
  if (1 < private_empty.size())
    ;
  if (private_empty.size() <= 1)
    ;
  if (1 >= private_empty.size())
    ;
  if (!private_empty.size())
    ;
  if (private_empty.size())
    ;

  // Types with weird size() return type.
  BoolSize bs;
  if (bs.size() == 0)
    ;
  EnumSize es;
  if (es.size() == 0)
    ;

  Derived derived;
  if (derived.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (derived.empty()){{$}}
  if (derived == Derived())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (derived.empty()){{$}}

  takesBool(derived.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}takesBool(!derived.empty());

  takesBool(derived.size() == 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}takesBool(derived.empty());

  takesBool(derived.size() != 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}takesBool(!derived.empty());

  bool b1 = derived.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}bool b1 = !derived.empty();

  bool b2(derived.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}bool b2(!derived.empty());

  auto b3 = static_cast<bool>(derived.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}auto b3 = static_cast<bool>(!derived.empty());

  auto b4 = (bool)derived.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}auto b4 = (bool)!derived.empty();

  auto b5 = bool(derived.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}auto b5 = bool(!derived.empty());

  return derived.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}return !derived.empty();
}

class ConstructWithBoolField {
  bool B;
public:
  ConstructWithBoolField(const std::vector<int> &C) : B(C.size()) {}
// CHECK-MESSAGES: :[[@LINE-1]]:57: warning: the 'empty' method should be used
// CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
// CHECK-FIXES: {{^  }}ConstructWithBoolField(const std::vector<int> &C) : B(!C.empty()) {}
};

struct StructWithNSDMI {
  std::vector<int> C;
  bool B = C.size();
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: the 'empty' method should be used
// CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
// CHECK-FIXES: {{^  }}bool B = !C.empty();
};

int func(const std::vector<int> &C) {
  return C.size() ? 0 : 1;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used
// CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
// CHECK-FIXES: {{^  }}return !C.empty() ? 0 : 1;
}

constexpr Lazy L;
static_assert(!L.size(), "");
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: the 'empty' method should be used
// CHECK-MESSAGES: :94:18: note: method 'Lazy'::empty() defined here
// CHECK-FIXES: {{^}}static_assert(L.empty(), "");

struct StructWithLazyNoexcept {
  void func() noexcept(L.size());
};

#define CHECKSIZE(x) if (x.size()) {}
// CHECK-FIXES: #define CHECKSIZE(x) if (x.size()) {}

template <typename T> void f() {
  std::vector<T> v;
  if (v.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!v.empty()){{$}}
  if (v == std::vector<T>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of comparing to an empty object [readability-container-size-empty]
  // CHECK-FIXES: {{^  }}if (v.empty()){{$}}
  // CHECK-FIXES-NEXT: ;
  CHECKSIZE(v);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: CHECKSIZE(v);

  TemplatedContainer<T> templated_container;
  if (templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (templated_container != TemplatedContainer<T>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  // CHECK-FIXES-NEXT: ;
  CHECKSIZE(templated_container);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: CHECKSIZE(templated_container);
  std::basic_string<T> s;
  if (s.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: string:{{[0-9]+}}:8: note: method 'basic_string'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!s.empty()){{$}}
  if (s.length())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'length' [readability-container-size-empty]
  // CHECK-MESSAGES: string:{{[0-9]+}}:8: note: method 'basic_string'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!s.empty()){{$}}
}

void g() {
  f<int>();
  f<double>();
  f<char *>();
}

template <typename T>
bool neverInstantiatedTemplate() {
  std::vector<T> v;
  if (v.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!v.empty()){{$}}

  if (v == std::vector<T>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of comparing to an empty object [readability-container-size-empty]
  // CHECK-FIXES: {{^  }}if (v.empty()){{$}}
  // CHECK-FIXES-NEXT: ;
  if (v.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}if (v.empty()){{$}}
  if (v.size() != 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!v.empty()){{$}}
  if (v.size() < 1)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}if (v.empty()){{$}}
  if (v.size() > 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!v.empty()){{$}}
  if (v.size() == 1)
    ;
  if (v.size() != 1)
    ;
  if (v.size() == 2)
    ;
  if (v.size() != 2)
    ;

  if (static_cast<bool>(v.size()))
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}if (static_cast<bool>(!v.empty())){{$}}
  if (v.size() && false)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!v.empty() && false){{$}}
  if (!v.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}if (v.empty()){{$}}

  TemplatedContainer<T> templated_container;
  if (templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  if (templated_container != TemplatedContainer<T>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}if (!templated_container.empty()){{$}}
  // CHECK-FIXES-NEXT: ;
  while (templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:10: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}while (!templated_container.empty()){{$}}

  do {
  }
  while (templated_container.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}while (!templated_container.empty());

  for (; templated_container.size();)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:10: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}for (; !templated_container.empty();){{$}}

  if (true && templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}if (true && !templated_container.empty()){{$}}

  if (true || templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}if (true || !templated_container.empty()){{$}}

  if (!templated_container.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}if (templated_container.empty()){{$}}

  bool b1 = templated_container.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}bool b1 = !templated_container.empty();

  bool b2(templated_container.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}bool b2(!templated_container.empty());

  auto b3 = static_cast<bool>(templated_container.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}auto b3 = static_cast<bool>(!templated_container.empty());

  auto b4 = (bool)templated_container.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}auto b4 = (bool)!templated_container.empty();

  auto b5 = bool(templated_container.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}auto b5 = bool(!templated_container.empty());

  takesBool(templated_container.size());
  // We don't detect this one because we don't know the parameter of takesBool
  // until the type of templated_container.size() is known

  return templated_container.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :37:8: note: method 'TemplatedContainer'::empty() defined here
  // CHECK-FIXES: {{^  }}return !templated_container.empty();
}

template <typename TypeRequiresSize>
void instantiatedTemplateWithSizeCall() {
  TypeRequiresSize t;
  // The instantiation of the template with std::vector<int> should not
  // result in changing the template, because we don't know that
  // TypeRequiresSize generally has `.empty()`
  if (t.size())
    ;

  if (t == TypeRequiresSize{})
    ;

  if (t != TypeRequiresSize{})
    ;
}

class TypeWithSize {
public:
  TypeWithSize();
  bool operator==(const TypeWithSize &other) const;
  bool operator!=(const TypeWithSize &other) const;

  unsigned size() const { return 0; }
  // Does not have `.empty()`
};

void instantiator() {
  instantiatedTemplateWithSizeCall<TypeWithSize>();
  instantiatedTemplateWithSizeCall<std::vector<int>>();
}

namespace std {
template <typename T>
struct unique_ptr {
  T *operator->() const;
  T &operator*() const;
};
} // namespace std

bool call_through_unique_ptr(const std::unique_ptr<std::vector<int>> &ptr) {
  return ptr->size() > 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}return !ptr->empty();
}

bool call_through_unique_ptr_deref(const std::unique_ptr<std::vector<int>> &ptr) {
  return (*ptr).size() > 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used
  // CHECK-MESSAGES: :12:8: note: method 'vector'::empty() defined here
  // CHECK-FIXES: {{^  }}return !(*ptr).empty();
}

struct TypedefSize {
  typedef int size_type;
  size_type size() const;
  bool empty() const;
};

void testTypedefSize() {
  TypedefSize ts;
  if (ts.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (ts.empty()){{$}}
}

namespace std {

template <typename T, unsigned long N> struct array {
  bool operator==(const array& other) const;
  bool operator!=(const array& other) const;
  unsigned long size() const { return N; }
  bool empty() const { return N != 0U; }

  T data[N];
};

}

struct DummyType {
  bool operator==(const DummyType&) const;
  unsigned long size() const;
  bool empty() const;
};

struct IgnoredDummyType {
  bool operator==(const IgnoredDummyType&) const;
  unsigned long size() const;
  bool empty() const;
};

typedef std::array<int, 10U> Array;

bool testArraySize(const Array& value) {
  return value.size() == 0U;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
// CHECK-FIXES: {{^  }}return value.empty();{{$}}
// CHECK-MESSAGES: :[[@LINE-25]]:8: note: method 'array'::empty() defined here
}

bool testArrayCompareToEmpty(const Array& value) {
  return value == std::array<int, 10U>();
}

bool testDummyType(const DummyType& value) {
  return value == DummyType();
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used to check for emptiness instead of comparing to an empty object [readability-container-size-empty]
// CHECK-FIXES: {{^  }}return value.empty();{{$}}
// CHECK-MESSAGES: :[[@LINE-26]]:8: note: method 'DummyType'::empty() defined here
}

bool testIgnoredDummyType(const IgnoredDummyType& value) {
  return value == IgnoredDummyType();
}

bool testStringLiterals(const std::string& s)
{
  using namespace std::string_literals;
  return s == ""s;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}return s.empty()
}

bool testNotEmptyStringLiterals(const std::string& s)
{
  using namespace std::string_literals;
  return s == "foo"s;
}

namespace PR72619 {
  struct SS {
    bool empty() const;
    int size() const;
  };

  struct SU {
    bool empty() const;
    unsigned size() const;
  };

  void f(const SU& s) {
    if (s.size() < 0) {}
    if (0 > s.size()) {}
    if (s.size() >= 0) {}
    if (0 <= s.size()) {}
    if (s.size() < 1)
      ;
    // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
    // CHECK-FIXES: {{^    }}if (s.empty()){{$}}
    if (1 > s.size())
      ;
    // CHECK-MESSAGES: :[[@LINE-2]]:13: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
    // CHECK-FIXES: {{^    }}if (s.empty()){{$}}
    if (s.size() <= 0)
      ;
    // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
    // CHECK-FIXES: {{^    }}if (s.empty()){{$}}
    if (0 >= s.size())
      ;
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
    // CHECK-FIXES: {{^    }}if (s.empty()){{$}}
  }

  void f(const SS& s) {
    if (s.size() < 0) {}
    if (0 > s.size()) {}
    if (s.size() >= 0) {}
    if (0 <= s.size()) {}
    if (s.size() < 1) {}
    if (1 > s.size()) {}
    if (s.size() <= 0) {}
    if (0 >= s.size()) {}
  }
}

namespace PR88203 {
  struct SS {
    bool empty() const;
    int size() const;
    int length(int) const;
  };

  struct SU {
    bool empty() const;
    int size(int) const;
    int length() const;
  };

  void f(const SS& s) {
    if (0 == s.length(1)) {}
    if (0 == s.size()) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
    // CHECK-FIXES: {{^    }}if (s.empty()) {}{{$}}
  }

  void f(const SU& s) {
    if (0 == s.size(1)) {}
    if (0 == s.length()) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: the 'empty' method should be used to check for emptiness instead of 'length' [readability-container-size-empty]
    // CHECK-FIXES: {{^    }}if (s.empty()) {}{{$}}
  }
}

namespace PR94454 {
  template <char...>
  int operator""_ci() { return 0; }
  auto eq = 0_ci == 0;
}

namespace GH152387 {

class foo : public std::string{
  void doit() {
    if (!size()) {
      // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used to check for emptiness instead of 'size'
      // CHECK-FIXES: if (empty()) {
    }
  }
};

}

class ReportInContainerNonEmptyMethod {
public:
  int size() const;
  bool empty() const;

  void doit() {
    if (!size()) {
      // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used to check for emptiness instead of 'size'
      // CHECK-FIXES: if (empty())
    }
  }
};

template <typename T>
class ReportInTemplateContainerNonEmptyMethod {
public:
  int size() const;
  bool empty() const;

  void doit() {
    if (!size()) {
      // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: the 'empty' method should be used to check for emptiness instead of 'size'
      // CHECK-FIXES: if (empty()) {
    }
  }
};



class ReportInContainerNonEmptyMethodCompare {
public:
  bool operator==(const ReportInContainerNonEmptyMethodCompare& other) const;
  int size() const;
  bool empty() const;

  void doit() {
    if (*this == ReportInContainerNonEmptyMethodCompare()) {
      // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: the 'empty' method should be used to check for emptiness instead of comparing to an empty object
      // CHECK-FIXES: if (this->empty()) {
    }
  }
};
