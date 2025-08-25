// RUN: %clangxx_tsan -fsanitize=undefined -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <iostream>
#include <thread>

class Foo {
public:
  void produce(int) {}
  void consume() {}

  void run() {
    w1_ = std::thread{&Foo::produce, this, 0};
    w2_ = std::thread{&Foo::consume, this};
    w1_.join();
    w2_.join();
  }

private:
  std::thread w1_;
  std::thread w2_;
};

int main() {
  Foo f;
  f.run();
  std::cerr << "Pass\n";
  // CHECK-NOT: data race
  // CHECK: Pass
  return 0;
}
