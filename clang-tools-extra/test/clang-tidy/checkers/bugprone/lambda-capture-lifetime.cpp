// RUN: %check_clang_tidy %s bugprone-lambda-capture-lifetime %t

namespace std {
class thread {
public:
  template <typename Callable> thread(Callable&& f) {}
};

template <typename T> class function {
public:
  function() = default;
  template <typename Callable> function(Callable&& f) {}
};

template <typename T>
class vector {
public:
  void emplace_back(T t) {}
  void push_back(T t) {}
};

template <typename Callable>
void async(Callable&& f) {}
} // namespace std

std::vector<std::function<int()>> GlobalFns;
std::function<int()> make_function(int);

void test_thread() {
  int x = 0;

  std::thread t1([&x]() { x = 1; });
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: lambda captures local variables by reference, but escapes the local scope

  std::thread t2([x]() { int y = x; });
}

void test_async_function() {
  int x = 0;

  std::async([&x]() { x = 1; });
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: lambda captures local variables by reference, but escapes the local scope
}

void test_vector_escape() {
  int y = 0;

  GlobalFns.emplace_back([&y]() -> int { return y; });
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: lambda captures local variables by reference, but escapes the local scope

  GlobalFns.push_back(std::function<int()>([&y]() -> int { return y; }));
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: lambda captures local variables by reference, but escapes the local scope
}

void test_safe_local_vector() {
  int z = 0;
  std::vector<std::function<int()>> LocalFns;

  LocalFns.emplace_back([&z]() -> int { return z; });
}

void test_nested_lambda_inside_argument_does_not_escape() {
  int q = 0;

  GlobalFns.emplace_back(make_function(([&q]() -> int { return q; })()));
}
