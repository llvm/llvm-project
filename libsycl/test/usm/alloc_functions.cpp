// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <tuple>

using namespace sycl;

constexpr size_t Align = 256;

struct alignas(Align) Aligned {
  int X;
};

int main() {
  queue Q;
  context Ctx = Q.get_context();
  device Dev = Q.get_device();

  auto Check = [&Q](size_t Alignment, auto AllocFn, int Line = __builtin_LINE(),
                    int Case = 0) {
    // First allocation might naturally be over-aligned. Do several of them to
    // do the verification;
    decltype(AllocFn()) Arr[10];
    for (auto *&Elem : Arr)
      Elem = AllocFn();
    for (auto *Ptr : Arr) {
      auto Addr = reinterpret_cast<uintptr_t>(Ptr);
      if ((Addr & (Alignment - 1)) != 0) {
        std::cout << "Failed at line " << Line << ", case " << Case
                  << std::endl;
        assert(false && "Not properly aligned!");
        break; // Reached only if asserts are disabled.
      }
    }
    for (auto *Ptr : Arr)
      free(Ptr, Q);
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
              (Check(Expected, Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  auto MDevice = [&](auto... Args) {
    return malloc_device(sizeof(std::max_align_t), Args...);
  };
  CheckAll(FAlign,
           std::tuple{[&]() { return MDevice(Q); },
                      [&]() { return MDevice(Dev, Ctx); },
                      [&]() { return MDevice(Q, property_list{}); },
                      [&]() { return MDevice(Dev, Ctx, property_list{}); }});

  auto ADevice = [&](auto... Args) {
    return aligned_alloc_device(Align, 1024, Args...);
  };

  CheckAll(Align, std::tuple{
                      [&]() { return ADevice(Q); },
                      [&]() { return ADevice(Dev, Ctx); },
                      [&]() { return ADevice(Q, property_list{}); },
                      [&]() { return ADevice(Dev, Ctx, property_list{}); },
                  });

  auto MHost = [&](auto... Args) {
    return malloc_host(sizeof(std::max_align_t), Args...);
  };

  CheckAll(FAlign,
           std::tuple{[&]() { return MHost(Q); }, [&]() { return MHost(Ctx); },
                      [&]() { return MHost(Q, property_list{}); },
                      [&]() { return MHost(Ctx, property_list{}); }});

  auto AHost = [&](auto... Args) {
    return aligned_alloc_host(Align, 1024, Args...);
  };

  CheckAll(Align, std::tuple{
                      [&]() { return AHost(Q); },
                      [&]() { return AHost(Ctx); },
                      [&]() { return AHost(Q, property_list{}); },
                      [&]() { return AHost(Ctx, property_list{}); },
                  });

  if (Dev.has(aspect::usm_shared_allocations)) {
    auto MShared = [&](auto... Args) {
      return malloc_shared(sizeof(std::max_align_t), Args...);
    };

    CheckAll(FAlign,
             std::tuple{[&]() { return MShared(Q); },
                        [&]() { return MShared(Dev, Ctx); },
                        [&]() { return MShared(Q, property_list{}); },
                        [&]() { return MShared(Dev, Ctx, property_list{}); }});

    auto AShared = [&](auto... Args) {
      return aligned_alloc_shared(Align, 1024, Args...);
    };
    CheckAll(Align, std::tuple{
                        [&]() { return AShared(Q); },
                        [&]() { return AShared(Dev, Ctx); },
                        [&]() { return AShared(Q, property_list{}); },
                        [&]() { return AShared(Dev, Ctx, property_list{}); },
                    });
  }

  auto TDevice = [&](auto... Args) {
    return malloc_device<Aligned>(1, Args...);
  };
  CheckAll(Align, std::tuple{[&]() { return TDevice(Q); },
                             [&]() { return TDevice(Dev, Ctx); }});

  auto TADevice = [&](auto... Args) {
    return aligned_alloc_device<Aligned>(Align, 1, Args...);
  };

  CheckAll(Align, std::tuple{[&]() { return TADevice(Q); },
                             [&]() { return TADevice(Dev, Ctx); }});

  auto THost = [&](auto... Args) { return malloc_host<Aligned>(1, Args...); };
  CheckAll(Align, std::tuple{[&]() { return THost(Q); },
                             [&]() { return THost(Ctx); }});

  auto TAHost = [&](auto... Args) {
    return aligned_alloc_host<Aligned>(Align, 1, Args...);
  };
  CheckAll(Align, std::tuple{[&]() { return TAHost(Q); },
                             [&]() { return TAHost(Ctx); }});

  if (Dev.has(aspect::usm_shared_allocations)) {
    auto TShared = [&](auto... Args) {
      return malloc_shared<Aligned>(1, Args...);
    };
    CheckAll(Align, std::tuple{[&]() { return TShared(Q); },
                               [&]() { return TShared(Dev, Ctx); }});
    auto TAShared = [&](auto... Args) {
      return aligned_alloc_shared<Aligned>(Align, 1, Args...);
    };
    CheckAll(Align, std::tuple{[&]() { return TAShared(Q); },
                               [&]() { return TAShared(Dev, Ctx); }});
  }

  auto Malloc = [&](auto... Args) {
    return malloc(sizeof(std::max_align_t), Args...);
  };

  CheckAll(
      FAlign,
      std::tuple{[&]() { return Malloc(Q, usm::alloc::host); },
                 [&]() { return Malloc(Dev, Ctx, usm::alloc::host); },
                 [&]() { return Malloc(Q, usm::alloc::host, property_list{}); },
                 [&]() {
                   return Malloc(Dev, Ctx, usm::alloc::host, property_list{});
                 }});

  auto AMalloc = [&](auto... Args) {
    return aligned_alloc(Align, 1024, Args...);
  };

  CheckAll(Align,
           std::tuple{
               [&]() { return AMalloc(Q, usm::alloc::host); },
               [&]() { return AMalloc(Dev, Ctx, usm::alloc::host); },
               [&]() { return AMalloc(Q, usm::alloc::host, property_list{}); },
               [&]() {
                 return AMalloc(Dev, Ctx, usm::alloc::host, property_list{});
               },
           });

  auto TMalloc = [&](auto... Args) { return malloc<Aligned>(1, Args...); };
  CheckAll(Align,
           std::tuple{[&]() { return TMalloc(Q, usm::alloc::host); },
                      [&]() { return TMalloc(Dev, Ctx, usm::alloc::host); }});

  auto TAMalloc = [&](auto... Args) {
    return aligned_alloc<Aligned>(Align, 1, Args...);
  };

  CheckAll(Align,
           std::tuple{[&]() { return TAMalloc(Q, usm::alloc::host); },
                      [&]() { return TAMalloc(Dev, Ctx, usm::alloc::host); }});

  // Testing invalid arguments for alignment
  assert(aligned_alloc_device(3, 1024, Q) == nullptr);
  assert(aligned_alloc_host(3, 1024, Q) == nullptr);
  assert(aligned_alloc_shared(3, 1024, Q) == nullptr);

  // A requested alignment of 0 means "no specific alignment" and must
  // succeed, routing through the plain (non-aligned) allocation path.
  void *ZeroAlignPtr = aligned_alloc_device(0, 1024, Q);
  assert(ZeroAlignPtr != nullptr);
  free(ZeroAlignPtr, Q);

  ZeroAlignPtr = aligned_alloc_host(0, 1024, Q);
  assert(ZeroAlignPtr != nullptr);
  free(ZeroAlignPtr, Q);

  ZeroAlignPtr = aligned_alloc_shared(0, 1024, Q);
  assert(ZeroAlignPtr != nullptr);
  free(ZeroAlignPtr, Q);

  return 0;
}
