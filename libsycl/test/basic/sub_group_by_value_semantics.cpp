// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  bool *Result = sycl::malloc_shared<bool>(1, Q);
  Result[0] = true;

  Q.parallel_for<class sub_group_by_value_semantics>(
      sycl::nd_range<3>({1, 1, 1}, {1, 1, 1}), [=](sycl::nd_item<3> Item) {
        sycl::sub_group A = Item.get_sub_group();

        // Check reflexivity.
        Result[0] &= (A == A);
        Result[0] &= !(A != A);

        // Check symmetry.
        auto Copied = A;
        auto &B = Copied;
        Result[0] &= (A == B);
        Result[0] &= (B == A);
        Result[0] &= !(A != B);
        Result[0] &= !(B != A);

        // Check transitivity.
        auto CopiedTwice = Copied;
        const auto &C = CopiedTwice;
        Result[0] &= (C == A);
      });

  Q.wait();

  bool Fail = !Result[0];
  sycl::free(Result, Q);
  return Fail;
}
