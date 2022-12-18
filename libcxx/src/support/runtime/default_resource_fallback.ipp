// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace std {

namespace pmr {

// default_memory_resource()

static memory_resource* __default_memory_resource(bool set = false, memory_resource* new_res = nullptr) noexcept {
#ifndef _LIBCPP_HAS_NO_ATOMIC_HEADER
  static constinit atomic<memory_resource*> __res{&res_init.resources.new_delete_res};
  if (set) {
    new_res = new_res ? new_res : new_delete_resource();
    // TODO: Can a weaker ordering be used?
    return std::atomic_exchange_explicit(&__res, new_res, memory_order_acq_rel);
  } else {
    return std::atomic_load_explicit(&__res, memory_order_acquire);
  }
#elif !defined(_LIBCPP_HAS_NO_THREADS)
  static constinit memory_resource* res = &res_init.resources.new_delete_res;
  static mutex res_lock;
  if (set) {
    new_res = new_res ? new_res : new_delete_resource();
    lock_guard<mutex> guard(res_lock);
    memory_resource* old_res = res;
    res = new_res;
    return old_res;
  } else {
    lock_guard<mutex> guard(res_lock);
    return res;
  }
#else
  static constinit memory_resource* res = &res_init.resources.new_delete_res;
  if (set) {
    new_res                  = new_res ? new_res : new_delete_resource();
    memory_resource* old_res = res;
    res                      = new_res;
    return old_res;
  } else {
    return res;
  }
#endif
}

memory_resource* get_default_resource() noexcept { return __default_memory_resource(); }

memory_resource* set_default_resource(memory_resource* __new_res) noexcept {
  return __default_memory_resource(true, __new_res);
}

// new_delete_resource()

#ifdef _LIBCPP_HAS_NO_ALIGNED_ALLOCATION
static bool is_aligned_to(void* ptr, size_t align) {
  void* p2     = ptr;
  size_t space = 1;
  void* result = std::align(align, 1, p2, space);
  return (result == ptr);
}
#endif

class _LIBCPP_TYPE_VIS __new_delete_memory_resource_imp : public memory_resource {
  void* do_allocate(size_t bytes, size_t align) override {
#ifndef _LIBCPP_HAS_NO_ALIGNED_ALLOCATION
    return std::__libcpp_allocate(bytes, align);
#else
    if (bytes == 0)
      bytes = 1;
    void* result = std::__libcpp_allocate(bytes, align);
    if (!is_aligned_to(result, align)) {
      std::__libcpp_deallocate(result, bytes, align);
      __throw_bad_alloc();
    }
    return result;
#endif
  }

  void do_deallocate(void* p, size_t bytes, size_t align) override { std::__libcpp_deallocate(p, bytes, align); }

  bool do_is_equal(const memory_resource& other) const noexcept override { return &other == this; }
};

// null_memory_resource()

class _LIBCPP_TYPE_VIS __null_memory_resource_imp : public memory_resource {
  void* do_allocate(size_t, size_t) override { __throw_bad_alloc(); }
  void do_deallocate(void*, size_t, size_t) override {}
  bool do_is_equal(const memory_resource& other) const noexcept override { return &other == this; }
};

namespace {

union ResourceInitHelper {
  struct {
    __new_delete_memory_resource_imp new_delete_res;
    __null_memory_resource_imp null_res;
  } resources;
  char dummy;
  _LIBCPP_CONSTEXPR_SINCE_CXX14 ResourceInitHelper() : resources() {}
  ~ResourceInitHelper() {}
};

// Pretend we're inside a system header so the compiler doesn't flag the use of the init_priority
// attribute with a value that's reserved for the implementation (we're the implementation).
#include "memory_resource_init_helper.h"

} // end namespace

memory_resource* new_delete_resource() noexcept { return &res_init.resources.new_delete_res; }

memory_resource* null_memory_resource() noexcept { return &res_init.resources.null_res; }

} // pmr
} // std
