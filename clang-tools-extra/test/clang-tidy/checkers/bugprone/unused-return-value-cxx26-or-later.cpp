// RUN: %check_clang_tidy -std=c++26-or-later %s bugprone-unused-return-value %t

namespace std {
struct future {};
template <typename Function, typename... Args>
future async(Function &&, Args &&...);
} // namespace std

int baz();

void disallowCastToVoidInCxx26() {
  (void)std::async(baz);
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  auto _ = std::async(baz);
}
