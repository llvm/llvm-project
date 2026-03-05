// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -fcxx-exceptions -fexceptions
// XFAIL: *
//
// std::make_exception_ptr crashes - exception handling NYI
// Related to exception system design

#include <exception>
#include <stdexcept>

void test() {
  std::exception_ptr ep = std::make_exception_ptr(std::runtime_error("test"));
}
