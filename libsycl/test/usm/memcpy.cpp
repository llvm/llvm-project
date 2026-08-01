// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cstddef>
#include <numeric>
#include <tuple>

using namespace sycl;

constexpr std::size_t DataSize = 1024;
constexpr std::size_t NumBytes = DataSize * sizeof(int);

// Tests memcpy API by making multiple memory allocations using AllocFs and
// performing a sequence of copies from one allocation to the next,
// using MemCpyFunc to specify dependencies.
// Assumes that the first and the last allocations are accessible on host.
template <typename MemcpyFuncT, typename... AllocFuncssT>
void test(queue &Q, MemcpyFuncT MemCpyFunc,
          std::tuple<AllocFuncssT...> AllocFs) {
  constexpr std::size_t NAllocations = std::tuple_size_v<decltype(AllocFs)>;
  static_assert(NAllocations > 1);

  std::vector<std::shared_ptr<int>> PtrPipeline;
  PtrPipeline.reserve(NAllocations);
  std::apply([&](auto &&...Fs) { ((PtrPipeline.push_back(Fs())), ...); },
             AllocFs);

  std::iota(PtrPipeline[0].get(), PtrPipeline[0].get() + DataSize, 0);

  // No dependencies for the first operation.
  event LastEvent =
      Q.memcpy(PtrPipeline[1].get(), PtrPipeline[0].get(), NumBytes);
  for (int I = 1; I < NAllocations - 1; ++I) {
    LastEvent = MemCpyFunc(Q, LastEvent, PtrPipeline[I + 1].get(),
                           PtrPipeline[I].get(), NumBytes);
  }

  Q.wait();

  int *ResultPtr = PtrPipeline.back().get();
  for (int I = 0; I < DataSize; ++I)
    assert(ResultPtr[I] == I);
}

template <bool ExplicitDeps, typename MemCpyFuncT>
void runTestsForMemCpyFunc(MemCpyFuncT MemCpyFunc) {
  // TODO: add in_order property here when ExplicitDeps == false once it is
  // implemented. For now all sycl::queue objects are in-order due to liboffload
  // limitation.
  sycl::queue Q;
  auto HostAllocF = [&]() {
    return std::shared_ptr<int>(new int[DataSize],
                                [&](int *Ptr) { delete[] Ptr; });
  };
  auto USMDeleter = [&](int *Ptr) { free(Ptr, Q); };
  auto HostUSMAllocF = [&]() {
    return std::shared_ptr<int>(malloc_host<int>(DataSize, Q), USMDeleter);
  };
  auto DeviceUSMAllocF = [&]() {
    return std::shared_ptr<int>(malloc_device<int>(DataSize, Q), USMDeleter);
  };
  auto SharedUSMAllocF = [&]() {
    return std::shared_ptr<int>(malloc_shared<int>(DataSize, Q), USMDeleter);
  };
  auto RunTest = [&](auto... AllocFuncs) {
    test(Q, MemCpyFunc, std::tuple(AllocFuncs...));
  };

  // These cases only do a single memcpy, so they would end up using just the
  // overload without dependencies regardless of what MemCpyFunc is.
  // TODO: Pass a default-constructed event as a dependency in these cases
  // instead when those are implemented.
  if constexpr (!ExplicitDeps) {
    // TODO: Remove try-catch once host-to-host copies are supported.
    try {
      RunTest(HostAllocF, HostAllocF);
      assert(false);
    } catch (const sycl::exception &e) {
      assert(e.code() == make_error_code(errc::feature_not_supported));
    }
    RunTest(HostAllocF, HostUSMAllocF);
    RunTest(HostAllocF, SharedUSMAllocF);

    RunTest(HostUSMAllocF, HostUSMAllocF);
    RunTest(HostUSMAllocF, HostAllocF);
    RunTest(HostUSMAllocF, SharedUSMAllocF);

    RunTest(SharedUSMAllocF, SharedUSMAllocF);
    RunTest(SharedUSMAllocF, HostAllocF);
    RunTest(SharedUSMAllocF, HostUSMAllocF);
  }

  RunTest(HostAllocF, DeviceUSMAllocF, HostAllocF);
  RunTest(SharedUSMAllocF, DeviceUSMAllocF, SharedUSMAllocF);
  RunTest(HostUSMAllocF, DeviceUSMAllocF, HostUSMAllocF);
  RunTest(HostAllocF, DeviceUSMAllocF, DeviceUSMAllocF, HostAllocF);
}

int main() {
  runTestsForMemCpyFunc<false>([&](queue &Q, event Dep, auto... OtherArgs) {
    (void)Dep;
    return Q.memcpy(OtherArgs...);
  });
  runTestsForMemCpyFunc<true>([&](queue &Q, event Dep, auto... OtherArgs) {
    return Q.memcpy(OtherArgs..., Dep);
  });
  runTestsForMemCpyFunc<true>([&](queue &Q, event Dep, auto... OtherArgs) {
    return Q.memcpy(OtherArgs..., std::vector<event>({Dep}));
  });

  return 0;
}
