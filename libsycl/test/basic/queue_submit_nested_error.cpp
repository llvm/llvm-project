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
      (void)cgh;
      q.submit([&](sycl::handler &nested) {
        nested.single_task<class NestedSubmitKernel>([]() {});
      });
    });
  } catch (const sycl::exception &E) {
    Thrown = true;
    const std::string Msg = E.what();
    assert(Msg.find("cannot be nested") != std::string::npos);
  }

  assert(Thrown && "Expected exception for nested queue.submit calls");
  return 0;
}
