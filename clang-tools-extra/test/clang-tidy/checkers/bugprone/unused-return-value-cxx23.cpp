// RUN: %check_clang_tidy -std=c++23 %s bugprone-unused-return-value %t

namespace std {
struct future {};
template <typename Function, typename... Args>
future async(Function &&, Args &&...);
} // namespace std

int foo();

void allowCastToVoidInCxx23() {
  std::async(foo);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  (void)std::async(foo);
  [[maybe_unused]] auto _ = std::async(foo);
}
