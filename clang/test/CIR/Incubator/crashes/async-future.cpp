// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -fcxx-exceptions -fexceptions
// XFAIL: *
//
// std::async/std::future crashes - exception handling NYI
// Related to exception system design

#include <future>

void test() {
  auto f = std::async(std::launch::async, []{ return 42; });
  int result = f.get();
}
