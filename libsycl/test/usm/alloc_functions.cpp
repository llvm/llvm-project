// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>
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
  // returned by the underlying runtime as of now.
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

  auto ADevice = [&](auto... args) {
    return aligned_alloc_device(Align, 1024, args...);
  };

  CheckAll(Align, std::tuple{
                      [&]() { return ADevice(q); },
                      [&]() { return ADevice(d, ctx); },
                      [&]() { return ADevice(q, property_list{}); },
                      [&]() { return ADevice(d, ctx, property_list{}); },
                  });

  auto MHost = [&](auto... args) {
    return malloc_host(sizeof(std::max_align_t), args...);
  };

  CheckAll(FAlign,
           std::tuple{[&]() { return MHost(q); }, [&]() { return MHost(ctx); },
                      [&]() { return MHost(q, property_list{}); },
                      [&]() { return MHost(ctx, property_list{}); }});

  auto AHost = [&](auto... args) {
    return aligned_alloc_host(Align, 1024, args...);
  };

  CheckAll(Align, std::tuple{
                      [&]() { return AHost(q); },
                      [&]() { return AHost(ctx); },
                      [&]() { return AHost(q, property_list{}); },
                      [&]() { return AHost(ctx, property_list{}); },
                  });

  if (d.has(aspect::usm_shared_allocations)) {
    auto MShared = [&](auto... args) {
      return malloc_shared(sizeof(std::max_align_t), args...);
    };

    CheckAll(FAlign,
             std::tuple{[&]() { return MShared(q); },
                        [&]() { return MShared(d, ctx); },
                        [&]() { return MShared(q, property_list{}); },
                        [&]() { return MShared(d, ctx, property_list{}); }});

    auto AShared = [&](auto... args) {
      return aligned_alloc_shared(Align, 1024, args...);
    };
    CheckAll(Align, std::tuple{
                        [&]() { return AShared(q); },
                        [&]() { return AShared(d, ctx); },
                        [&]() { return AShared(q, property_list{}); },
                        [&]() { return AShared(d, ctx, property_list{}); },
                    });
  }

  auto TDevice = [&](auto... args) {
    return malloc_device<Aligned>(1, args...);
  };
  CheckAll(Align, std::tuple{[&]() { return TDevice(q); },
                             [&]() { return TDevice(d, ctx); }});

  auto TADevice = [&](auto... args) {
    return aligned_alloc_device<Aligned>(Align, 1, args...);
  };

  CheckAll(Align, std::tuple{[&]() { return TADevice(q); },
                             [&]() { return TADevice(d, ctx); }});

  auto THost = [&](auto... args) { return malloc_host<Aligned>(1, args...); };
  CheckAll(Align, std::tuple{[&]() { return THost(q); },
                             [&]() { return THost(ctx); }});

  auto TAHost = [&](auto... args) {
    return aligned_alloc_host<Aligned>(Align, 1, args...);
  };
  CheckAll(Align, std::tuple{[&]() { return TAHost(q); },
                             [&]() { return TAHost(ctx); }});

  if (d.has(aspect::usm_shared_allocations)) {
    auto TShared = [&](auto... args) {
      return malloc_shared<Aligned>(1, args...);
    };
    CheckAll(Align, std::tuple{[&]() { return TShared(q); },
                               [&]() { return TShared(d, ctx); }});
    auto TAShared = [&](auto... args) {
      return aligned_alloc_shared<Aligned>(Align, 1, args...);
    };
    CheckAll(Align, std::tuple{[&]() { return TAShared(q); },
                               [&]() { return TAShared(d, ctx); }});
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

  auto AMalloc = [&](auto... args) {
    return aligned_alloc(Align, 1024, args...);
  };

  CheckAll(
      Align,
      std::tuple{
          [&]() { return AMalloc(q, usm::alloc::host); },
          [&]() { return AMalloc(d, ctx, usm::alloc::host); },
          [&]() { return AMalloc(q, usm::alloc::host, property_list{}); },
          [&]() { return AMalloc(d, ctx, usm::alloc::host, property_list{}); },
      });

  auto TMalloc = [&](auto... args) { return malloc<Aligned>(1, args...); };
  CheckAll(Align,
           std::tuple{[&]() { return TMalloc(q, usm::alloc::host); },
                      [&]() { return TMalloc(d, ctx, usm::alloc::host); }});

  auto TAMalloc = [&](auto... args) {
    return aligned_alloc<Aligned>(Align, 1, args...);
  };

  CheckAll(Align,
           std::tuple{[&]() { return TAMalloc(q, usm::alloc::host); },
                      [&]() { return TAMalloc(d, ctx, usm::alloc::host); }});

  // Testing invalid arguments for alignment
  assert(aligned_alloc_device(3, 1024, q) == nullptr);
  assert(aligned_alloc_host(3, 1024, q) == nullptr);
  if (d.has(aspect::usm_shared_allocations))
    assert(aligned_alloc_shared(3, 1024, q) == nullptr);

  // A requested alignment of 0 means "no specific alignment" and must
  // succeed, routing through the plain (non-aligned) allocation path.
  void *ZeroAlignPtr = aligned_alloc_device(0, 1024, q);
  assert(ZeroAlignPtr != nullptr);
  free(ZeroAlignPtr, q);

  ZeroAlignPtr = aligned_alloc_host(0, 1024, q);
  assert(ZeroAlignPtr != nullptr);
  free(ZeroAlignPtr, q);

  if (d.has(aspect::usm_shared_allocations)) {
    ZeroAlignPtr = aligned_alloc_shared(0, 1024, q);
    assert(ZeroAlignPtr != nullptr);
    free(ZeroAlignPtr, q);
  }

  return 0;
}
