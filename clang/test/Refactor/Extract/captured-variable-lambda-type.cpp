
int capturedLambda(int x) {
  auto Lambda = [] () { };
  Lambda();
  auto Lambda2 = [] (int x, int y) -> int { return x + y * 2; };
  int y = Lambda2(x, 1);
  auto Lambda3 = [&] (int y) {
    x = y + 2;
  };
  Lambda3(3);
  return Lambda2(x, 1);
}

// CHECK1: extracted(const std::function<void ()> &Lambda)
// CHECK1: extracted(const std::function<auto (int, int) -> int> &Lambda2, int x)
// CHECK1: extracted(const std::function<auto (int, int) -> int> &Lambda2, const std::function<void (int)> &Lambda3, int x)

// RUN: clang-refactor-test perform -action extract -selected=%s:4:3-4:11 -selected=%s:6:3-6:24 -selected=%s:10:3-11:23 %s -std=c++11 | FileCheck --check-prefix=CHECK1 %s
