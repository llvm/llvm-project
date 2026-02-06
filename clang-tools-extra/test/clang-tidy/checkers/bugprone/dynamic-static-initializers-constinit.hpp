// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-dynamic-static-initializers %t -- -- -fno-threadsafe-statics -fno-delayed-template-parsing

constexpr int const_func() { return 42; }

constinit int a = const_func(); // no warning

constinit int b = 123; // no warning

struct S {
  static constinit int c;
};
constinit int S::c = const_func(); // no warning

int runtime_func() { return 42; }

int e = runtime_func();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: static variable 'e' may be dynamically initialized in this header file [bugprone-dynamic-static-initializers]

template <typename T>
struct TemplateS {
  static constinit int x;
};
template <typename T>
constinit int TemplateS<T>::x = const_func(); // no warning

template <typename T>
void template_func() {
  static constinit int v = const_func(); // no warning
}

void call_template_func() {
  template_func<int>();
}

template <int V>
struct Value {
  static constinit int v;
};
template <int V>
constinit int Value<V>::v = V; // no warning

thread_local constinit int tl = const_func(); // no warning

struct InlineS {
  static inline constinit int i = 42; // no warning
};

auto lambda = []() {
  static constinit int l = const_func(); // no warning
  return l;
};

struct Separate {
  static constinit int s;
};
constinit int Separate::s = 100; // no warning

extern int late_constinit;
constinit int late_constinit = 42; // no warning
