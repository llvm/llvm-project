// REQUIRES: any-device
// RUN: %clangxx -fsycl -Wno-error=deprecated-declarations %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>
#include <memory>

using namespace sycl;

// TODO: original test works with buffers, revert changes to USM once they are
// implemented.
int main() {
  bool Fail{};

  constexpr size_t DataSize = 10;
  const range<1> GlobalRange(6);
  // Id indexer
  {
    queue Q;
    int *Data = sycl::malloc_shared<int>(DataSize, Q);
    for (size_t i = 0; i < DataSize; ++i)
      Data[i] = -1;

    Q.parallel_for<class id1>(GlobalRange,
                              [=](id<1> Index) { Data[Index] = Index[0]; });
    Q.wait();

    Fail |= [&]() {
      for (size_t i = 0; i < DataSize; ++i) {
        const int ExpectedVal = i < GlobalRange[0] ? i : -1;
        if (Data[i] != ExpectedVal) {
          std::cout << "line: " << __LINE__ << " Data[" << i << "] is "
                    << Data[i] << " expected " << ExpectedVal << std::endl;
          return true;
        }
      }
      return false;
    }();

    free(Data, Q);
  }

  // Item indexer without offset
  {
    // TODO: replace strcut with sycl::int2 once implemented.
    struct DoubleInt {
      int Id;
      int Range;
    };
    queue Q;
    DoubleInt *Data = sycl::malloc_shared<DoubleInt>(DataSize, Q);
    for (size_t i = 0; i < DataSize; ++i)
      Data[i] = {-1, -1};

    Q.parallel_for<class item1_nooffset>(
        GlobalRange, [=](item<1, false> Index) {
          Data[Index.get_id()] = {int(Index.get_id()[0]),
                                  int(Index.get_range()[0])};
        });
    Q.wait();

    Fail |= [&]() {
      for (size_t i = 0; i < DataSize; ++i) {
        const int ExpectedValID = i < GlobalRange[0] ? i : -1;
        const int ExpectedValRange = i < GlobalRange[0] ? GlobalRange[0] : -1;
        if (Data[i].Id != ExpectedValID || Data[i].Range != ExpectedValRange) {
          std::cout << "line: " << __LINE__ << " Data[" << i << "] is {"
                    << Data[i].Id << ", " << Data[i].Range << "} expected {"
                    << ExpectedValID << ", " << ExpectedValRange << "}"
                    << std::endl;
          return true;
        }
      }
      return false;
    }();
    free(Data, Q);
  }

  // TODO:  Item indexer with offset
  // blocked by liboffload support
  // blocked by absence of sycl::handler implementation

  // TODO: add nd_item check
  return Fail;
}
