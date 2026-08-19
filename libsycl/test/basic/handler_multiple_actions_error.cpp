// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>
#include <string>

int main() {
  sycl::queue q;

  bool Thrown = false;
  try {
    q.submit([&](sycl::handler &cgh) {
      cgh.single_task<class FirstAction>([]() {});
      cgh.single_task<class SecondAction>([]() {});
    });
  } catch (const sycl::exception &E) {
    Thrown = true;
    const std::string Msg = E.what();
    assert(Msg.find("multiple actions") != std::string::npos);
  }

  assert(Thrown && "Expected exception for multiple command-group actions");
  return 0;
}
