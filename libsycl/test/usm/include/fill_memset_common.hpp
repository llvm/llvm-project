#include <sycl/sycl.hpp>

#include <array>
#include <cassert>

using namespace sycl;

constexpr std::size_t DataSize = 1024;
constexpr int Pattern = 42;

template <typename DataT> bool verify(DataT *Ptr) {
  for (int I = 0; I < DataSize; ++I)
    if (Ptr[I] != Pattern)
      return false;
  return true;
}

template <bool VerifyOnDevice, typename DataT, typename OpT>
void test(queue &Q, DataT *Ptr, OpT Op) {
  Op(Ptr);
  Q.wait();

  if constexpr (VerifyOnDevice) {
    bool *Result = malloc_shared<bool>(1, Q);
    Q.single_task<class Verify>([=]() { verify(Ptr); });
    Q.wait();
    assert(Result);
    sycl::free(Result, Q);
  } else {
    Q.wait();
    assert(verify(Ptr));
  }
  sycl::free(Ptr, Q);
}

template <typename DataT, typename OpT> void runTests(queue &Q, OpT Op) {
  test<false>(Q, malloc_host<DataT>(1024, Q), Op);
  test<true>(Q, malloc_host<DataT>(1024, Q), Op);
  test<false>(Q, malloc_shared<DataT>(1024, Q), Op);
  test<true>(Q, malloc_shared<DataT>(1024, Q), Op);
  test<true>(Q, malloc_device<DataT>(1024, Q), Op);
}