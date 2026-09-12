#include <sycl/sycl.hpp>

#include <cassert>

constexpr std::size_t DataSize = 1024;

template <typename DataT> bool verify(DataT *Ptr, int Pattern) {
  for (std::size_t I = 0; I < DataSize; ++I)
    if (Ptr[I] != static_cast<DataT>(Pattern))
      return false;
  return true;
}

template <bool VerifyOnDevice, typename DataT, typename OpT>
void test(sycl::queue &Q, DataT *Ptr, OpT Op, int Pattern) {
  Op(Ptr, Pattern);
  Q.wait();

  if constexpr (VerifyOnDevice) {
    bool *Result = sycl::malloc_shared<bool>(1, Q);
    Q.single_task<class Verify>([=]() { *Result = verify(Ptr, Pattern); });
    Q.wait();
    assert(*Result);
    sycl::free(Result, Q);
  } else {
    assert(verify(Ptr, Pattern));
  }
  sycl::free(Ptr, Q);
}

template <typename DataT, typename OpT>
void runTests(sycl::queue &Q, OpT Op, int Pattern = 42) {
  test<false>(Q, sycl::malloc_host<DataT>(DataSize, Q), Op, Pattern);
  test<true>(Q, sycl::malloc_host<DataT>(DataSize, Q), Op, Pattern);
  test<false>(Q, sycl::malloc_shared<DataT>(DataSize, Q), Op, Pattern);
  test<true>(Q, sycl::malloc_shared<DataT>(DataSize, Q), Op, Pattern);
  test<true>(Q, sycl::malloc_device<DataT>(DataSize, Q), Op, Pattern);
}
