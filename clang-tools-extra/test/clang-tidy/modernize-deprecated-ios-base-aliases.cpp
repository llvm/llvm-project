// RUN: %check_clang_tidy %s modernize-deprecated-ios-base-aliases %t

namespace std {
class ios_base {
public:
  typedef int io_state;
  typedef int open_mode;
  typedef int seek_dir;

  typedef int streampos;
  typedef int streamoff;
};

template <class CharT>
class basic_ios : public ios_base {
};
} // namespace std

// Test function return values (declaration)
std::ios_base::io_state f_5();
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'std::ios_base::io_state' is deprecated; use 'std::ios_base::iostate' instead [modernize-deprecated-ios-base-aliases]
// CHECK-FIXES: std::ios_base::iostate f_5();

// Test function parameters.
void f_6(std::ios_base::open_mode);
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 'std::ios_base::open_mode' is deprecated
// CHECK-FIXES: void f_6(std::ios_base::openmode);
void f_7(const std::ios_base::seek_dir &);
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 'std::ios_base::seek_dir' is deprecated
// CHECK-FIXES: void f_7(const std::ios_base::seekdir &);

// Test on record type fields.
struct A {
  std::ios_base::io_state field;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: std::ios_base::iostate field;

  typedef std::ios_base::io_state int_ptr_type;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: typedef std::ios_base::iostate int_ptr_type;
};

struct B : public std::ios_base {
  io_state a;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: iostate a;
};

struct C : public std::basic_ios<char> {
  io_state a;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: iostate a;
};

void f_1() {
  std::ios_base::io_state a;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: std::ios_base::iostate a;

  // Check that spaces aren't modified unnecessarily.
  std :: ios_base :: io_state b;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: std :: ios_base :: iostate b;

  // Test construction from a temporary.
  std::ios_base::io_state c = std::ios_base::io_state{};
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-MESSAGES: :[[@LINE-2]]:46: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: std::ios_base::iostate c = std::ios_base::iostate{};

  typedef std::ios_base::io_state alias1;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: typedef std::ios_base::iostate alias1;
  alias1 d(a);

  using alias2 = std::ios_base::io_state;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: using alias2 = std::ios_base::iostate;
  alias2 e;

  // Test pointers.
  std::ios_base::io_state *f;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: std::ios_base::iostate *f;

  // Test 'static' declarations.
  static std::ios_base::io_state g;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: static std::ios_base::iostate g;

  // Test with cv-qualifiers.
  const std::ios_base::io_state h(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: const std::ios_base::iostate h(0);
  volatile std::ios_base::io_state i;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: volatile std::ios_base::iostate i;
  const volatile std::ios_base::io_state j(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: const volatile std::ios_base::iostate j(0);

  // Test auto and initializer-list.
  auto k = std::ios_base::io_state{};
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: auto k = std::ios_base::iostate{};

  std::ios_base::io_state l{std::ios_base::io_state()};
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-MESSAGES: :[[@LINE-2]]:44: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: std::ios_base::iostate l{std::ios_base::iostate()};

  // Test temporaries.
  std::ios_base::io_state();
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: std::ios_base::iostate();

  // Test inherited type usage
  std::basic_ios<char>::io_state m;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: std::basic_ios<char>::iostate m;

  std::ios_base::streampos n;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'std::ios_base::streampos' is deprecated [modernize-deprecated-ios-base-aliases]

  std::ios_base::streamoff o;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'std::ios_base::streamoff' is deprecated [modernize-deprecated-ios-base-aliases]
}

// Test without the nested name specifiers.
void f_2() {
  using namespace std;

  ios_base::io_state a;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: ios_base::iostate a;
}

// Test messing-up with macros.
void f_4() {
#define MACRO_1 std::ios_base::io_state
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: 'std::ios_base::io_state' is deprecated
  MACRO_1 a;

#define MACRO_2 io_state
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'std::ios_base::io_state' is deprecated
  std::ios_base::MACRO_2 b;

#define MACRO_3 std::ios_base
  MACRO_3::io_state c;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'std::ios_base::io_state' is deprecated

#define MACRO_4(type) type::io_state
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: 'std::ios_base::io_state' is deprecated
  MACRO_4(std::ios_base) d;

#undef MACRO_1
#undef MACRO_2
#undef MACRO_3
#undef MACRO_4
}

// Test function return values (definition).
std::ios_base::io_state f_5()
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'std::ios_base::io_state' is deprecated
// CHECK-FIXES: std::ios_base::iostate f_5()
{
  // Test constructor.
  return std::ios_base::io_state(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: return std::ios_base::iostate(0);
}

// Test that other aliases with same name aren't replaced
struct my_ios_base {
  typedef int io_state;
};

namespace ns_1 {
struct my_ios_base2 {
  typedef int io_state;
};
} // namespace ns_1

void f_8() {
  my_ios_base::io_state a;

  ns_1::my_ios_base2::io_state b;
}

// Test templates
template <typename X>
void f_9() {
  typename std::basic_ios<X>::io_state p;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 'std::ios_base::io_state' is deprecated
  typename std::ios_base::io_state q;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: typename std::ios_base::iostate q;
}

template <typename T>
void f_10(T arg) {
  T x(arg);
}

template <typename T>
void f_11() {
  typename T::io_state x{};
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 'std::ios_base::io_state' is deprecated
}

template <typename T>
struct D : std::ios_base {
  io_state a;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: iostate a;

  typename std::basic_ios<T>::io_state b;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 'std::ios_base::io_state' is deprecated
};

template <typename T>
struct E {
  T t;
};

void f_12() {
  f_9<char>();

  f_10<std::ios_base::io_state>(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: f_10<std::ios_base::iostate>(0);

  f_11<std::ios_base>();
  D<char> d;

  E<std::ios_base::io_state> e;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 'std::ios_base::io_state' is deprecated
  // CHECK-FIXES: E<std::ios_base::iostate> e;
}
