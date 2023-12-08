// RUN: %check_clang_tidy %s boost-use-to-string %t

namespace std {

template <typename T>
class basic_string {};

using string = basic_string<char>;
using wstring = basic_string<wchar_t>;
}

namespace boost {
template <typename T, typename V>
T lexical_cast(const V &) {
  return T();
};
}

struct my_weird_type {};

std::string fun(const std::string &) {}

void test_to_string1() {

  auto xa = boost::lexical_cast<std::string>(5);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::to_string instead of boost::lexical_cast<std::string> [boost-use-to-string]
  // CHECK-FIXES: auto xa = std::to_string(5);

  auto z = boost::lexical_cast<std::string>(42LL);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use std::to_string
  // CHECK-FIXES: auto z = std::to_string(42LL);

  // this should not trigger
  fun(boost::lexical_cast<std::string>(42.0));
  auto non = boost::lexical_cast<my_weird_type>(42);
  boost::lexical_cast<int>("12");
}

void test_to_string2() {
  int a;
  long b;
  long long c;
  unsigned d;
  unsigned long e;
  unsigned long long f;
  float g;
  double h;
  long double i;
  bool j;

  fun(boost::lexical_cast<std::string>(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_string
  // CHECK-FIXES: fun(std::to_string(a));
  fun(boost::lexical_cast<std::string>(b));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_string
  // CHECK-FIXES: fun(std::to_string(b));
  fun(boost::lexical_cast<std::string>(c));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_string
  // CHECK-FIXES: fun(std::to_string(c));
  fun(boost::lexical_cast<std::string>(d));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_string
  // CHECK-FIXES: fun(std::to_string(d));
  fun(boost::lexical_cast<std::string>(e));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_string
  // CHECK-FIXES: fun(std::to_string(e));
  fun(boost::lexical_cast<std::string>(f));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_string
  // CHECK-FIXES: fun(std::to_string(f));

  // No change for floating numbers.
  fun(boost::lexical_cast<std::string>(g));
  fun(boost::lexical_cast<std::string>(h));
  fun(boost::lexical_cast<std::string>(i));
  // And bool.
  fun(boost::lexical_cast<std::string>(j));
}

std::string fun(const std::wstring &) {}

void test_to_wstring() {
  int a;
  long b;
  long long c;
  unsigned d;
  unsigned long e;
  unsigned long long f;
  float g;
  double h;
  long double i;
  bool j;

  fun(boost::lexical_cast<std::wstring>(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_wstring instead of boost::lexical_cast<std::wstring> [boost-use-to-string]
  // CHECK-FIXES: fun(std::to_wstring(a));
  fun(boost::lexical_cast<std::wstring>(b));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_wstring
  // CHECK-FIXES: fun(std::to_wstring(b));
  fun(boost::lexical_cast<std::wstring>(c));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_wstring
  // CHECK-FIXES: fun(std::to_wstring(c));
  fun(boost::lexical_cast<std::wstring>(d));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_wstring
  // CHECK-FIXES: fun(std::to_wstring(d));
  fun(boost::lexical_cast<std::wstring>(e));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_wstring
  // CHECK-FIXES: fun(std::to_wstring(e));
  fun(boost::lexical_cast<std::wstring>(f));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::to_wstring
  // CHECK-FIXES: fun(std::to_wstring(f));

  // No change for floating numbers
  fun(boost::lexical_cast<std::wstring>(g));
  fun(boost::lexical_cast<std::wstring>(h));
  fun(boost::lexical_cast<std::wstring>(i));
  // and bool.
  fun(boost::lexical_cast<std::wstring>(j));
}

const auto glob = boost::lexical_cast<std::string>(42);
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use std::to_string
// CHECK-FIXES: const auto glob = std::to_string(42);

template <typename T>
void string_as_T(T t = T()) {
  boost::lexical_cast<std::string>(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use std::to_string
  // CHECK-FIXES: std::to_string(42);

  boost::lexical_cast<T>(42);
  string_as_T(boost::lexical_cast<T>(42));
  auto p = boost::lexical_cast<T>(42);
  auto p2 = (T)boost::lexical_cast<T>(42);
  auto p3 = static_cast<T>(boost::lexical_cast<T>(42));
}

#define my_to_string boost::lexical_cast<std::string>

void no_fixup_inside_macro() {
  my_to_string(12);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use std::to_string
}

void no_warnings() {
  fun(boost::lexical_cast<std::string>("abc"));
  fun(boost::lexical_cast<std::wstring>("abc"));
  fun(boost::lexical_cast<std::string>(my_weird_type{}));
  string_as_T<int>();
  string_as_T<std::string>();
}

struct Fields {
  int integer;
  float floating;
  Fields* wierd;
  const int &getConstInteger() const {return integer;}
};

void testFields() {
  Fields fields;
  auto s1 = boost::lexical_cast<std::string>(fields.integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::to_string
  // CHECK-FIXES: auto s1 = std::to_string(fields.integer);

  auto s2 = boost::lexical_cast<std::string>(fields.floating);
  auto s3 = boost::lexical_cast<std::string>(fields.wierd);
  auto s4 = boost::lexical_cast<std::string>(fields.getConstInteger());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::to_string
  // CHECK-FIXES: auto s4 = std::to_string(fields.getConstInteger());
}
