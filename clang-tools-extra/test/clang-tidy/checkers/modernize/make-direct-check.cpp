// RUN: %check_clang_tidy %s modernize-make-direct %t

namespace std {
template<typename T>
struct optional { 
 optional(const T&) {} 
};

template<typename T> 
optional<T> make_optional(const T& t) { return optional<T>(t); }

template<typename T>
struct unique_ptr {
 explicit unique_ptr(T*) {}
};

template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) { 
 return unique_ptr<T>(new T(args...));
}

template<typename T>
struct shared_ptr {
 shared_ptr(T*) {}
};

template<typename T, typename... Args>
shared_ptr<T> make_shared(Args&&... args) {
 return shared_ptr<T>(new T(args...));
}

template<typename T, typename U>
struct pair {
 T first;
 U second;
 pair(const T& x, const U& y) : first(x), second(y) {}
};

template<typename T, typename U>
pair<T,U> make_pair(T&& t, U&& u) { 
 return pair<T,U>(t, u); 
}
}

struct Widget {
 Widget(int x) {}
};


void basic_tests() {
 auto o1 = std::make_optional<int>(42);
 // CHECK-MESSAGES: warning: use class template argument deduction (CTAD) instead of make_optional [modernize-make-direct]
 // CHECK-FIXES: auto o1 = std::optional(42);

  auto u1 = std::make_unique<Widget>(1);
 // CHECK-MESSAGES: warning: use class template argument deduction (CTAD) instead of make_unique [modernize-make-direct]
 // CHECK-FIXES: auto u1 = std::unique_ptr(new Widget(1));

 auto s1 = std::make_shared<Widget>(2);
  // CHECK-MESSAGES: warning: use class template argument deduction (CTAD) instead of make_shared
  // CHECK-FIXES: auto s1 = std::shared_ptr(new Widget(2));

  auto p1 = std::make_pair(1, "test");
  // CHECK-MESSAGES: warning: use class template argument deduction (CTAD) instead of make_pair
  // CHECK-FIXES: auto p1 = std::pair(1, "test");
}
