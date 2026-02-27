// REQUIRES: any-device
// RUN: %clangxx %sycl_options %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cstddef>
#include <iostream>
#include <tuple>

using namespace sycl;

constexpr size_t Align = 256;

struct alignas(Align) Aligned {
  int x;
};

int main() {
  queue q;
  context ctx = q.get_context();
  device d = q.get_device();

  auto check = [&q](size_t Alignment, auto AllocFn, int Line = __builtin_LINE(),
                    int Case = 0) {
    // First allocation might naturally be over-aligned. Do several of them to
    // do the verification;
    decltype(AllocFn()) Arr[10];
    for (auto *&Elem : Arr)
      Elem = AllocFn();
    for (auto *Ptr : Arr) {
      auto v = reinterpret_cast<uintptr_t>(Ptr);
      if ((v & (Alignment - 1)) != 0) {
        std::cout << "Failed at line " << Line << ", case " << Case
                  << std::endl;
        assert(false && "Not properly aligned!");
        break; // To be used with commented out assert above.
      }
    }
    for (auto *Ptr : Arr)
      free(Ptr, q);
  };

  // The strictest (largest) fundamental alignment of any type is the alignment
  // of max_align_t. This is, however, smaller than the minimal alignment
  // returned by the underlyging runtime as of now.
  constexpr size_t FAlign = alignof(std::max_align_t);

  auto CheckAll = [&](size_t Expected, auto Funcs,
                      int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0;
          (void)std::initializer_list<int>{
              (check(Expected, Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  auto MDevice = [&](auto... args) {
    return malloc_device(sizeof(std::max_align_t), args...);
  };
  CheckAll(FAlign,
           std::tuple{[&]() { return MDevice(q); },
                      [&]() { return MDevice(d, ctx); },
                      [&]() { return MDevice(q, property_list{}); },
                      [&]() { return MDevice(d, ctx, property_list{}); }});

  auto MHost = [&](auto... args) {
    return malloc_host(sizeof(std::max_align_t), args...);
  };
  CheckAll(FAlign,
           std::tuple{[&]() { return MHost(q); }, [&]() { return MHost(ctx); },
                      [&]() { return MHost(q, property_list{}); },
                      [&]() { return MHost(ctx, property_list{}); }});

  if (d.has(aspect::usm_shared_allocations)) {
    auto MShared = [&](auto... args) {
      return malloc_shared(sizeof(std::max_align_t), args...);
    };

    CheckAll(FAlign,
             std::tuple{[&]() { return MShared(q); },
                        [&]() { return MShared(d, ctx); },
                        [&]() { return MShared(q, property_list{}); },
                        [&]() { return MShared(d, ctx, property_list{}); }});
  }

  auto TDevice = [&](auto... args) {
    return malloc_device<Aligned>(1, args...);
  };
  CheckAll(Align, std::tuple{[&]() { return TDevice(q); },
                             [&]() { return TDevice(d, ctx); }});

  auto THost = [&](auto... args) { return malloc_host<Aligned>(1, args...); };
  CheckAll(Align, std::tuple{[&]() { return THost(q); },
                             [&]() { return THost(ctx); }});

  if (d.has(aspect::usm_shared_allocations)) {
    auto TShared = [&](auto... args) {
      return malloc_shared<Aligned>(1, args...);
    };
    CheckAll(Align, std::tuple{[&]() { return TShared(q); },
                               [&]() { return TShared(d, ctx); }});
  }

  auto Malloc = [&](auto... args) {
    return malloc(sizeof(std::max_align_t), args...);
  };
  CheckAll(
      FAlign,
      std::tuple{
          [&]() { return Malloc(q, usm::alloc::host); },
          [&]() { return Malloc(d, ctx, usm::alloc::host); },
          [&]() { return Malloc(q, usm::alloc::host, property_list{}); },
          [&]() { return Malloc(d, ctx, usm::alloc::host, property_list{}); }});

  auto TMalloc = [&](auto... args) { return malloc<Aligned>(1, args...); };
  CheckAll(Align,
           std::tuple{[&]() { return TMalloc(q, usm::alloc::host); },
                      [&]() { return TMalloc(d, ctx, usm::alloc::host); }});

  return 0;
}
