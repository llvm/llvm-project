//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <map>
#include <mutex>
#include <shared_mutex>

_LIBCPP_BEGIN_NAMESPACE_STD

class __shared_resource {
public:
  _LIBCPP_HIDE_FROM_ABI __shared_resource()              = default;
  __shared_resource(const __shared_resource&)            = delete;
  __shared_resource& operator=(const __shared_resource&) = delete;

  _LIBCPP_HIDE_FROM_ABI mutex& __inc_reference(const void* __ptr) {
    _LIBCPP_ASSERT_NON_NULL(__ptr != nullptr, "not a valid resource");
    unique_lock __lock{__mutex_};

    auto& __resource = __lut_[reinterpret_cast<uintptr_t>(__ptr)];
    ++__resource.__count;
    return __resource.__mutex;
  }

  _LIBCPP_HIDE_FROM_ABI void __dec_reference(const void* __ptr) {
    unique_lock __lock{__mutex_};

    auto __it = __get_it(__ptr);
    if (__it->second.__count == 1)
      __lut_.erase(__it);
    else
      --__it->second.__count;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI lock_guard<mutex> __get_lock(const void* __ptr) {
    shared_lock __lock{__mutex_};
    return lock_guard{__get_it(__ptr)->second.__mutex};
  }

  [[nodiscard]] static _LIBCPP_HIDE_FROM_ABI __shared_resource& __instance() {
    static __shared_resource __result;
    return __result;
  }

private:
  struct __value {
    mutex __mutex;
    size_t __count{0};
  };

  shared_mutex __mutex_;
  map<uintptr_t, __value> __lut_;

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI map<uintptr_t, __value>::iterator __get_it(const void* __ptr) {
    _LIBCPP_ASSERT_NON_NULL(__ptr != nullptr, "not a valid resource");

    auto __it = __lut_.find(reinterpret_cast<uintptr_t>(__ptr));
    _LIBCPP_ASSERT_INTERNAL(__it != __lut_.end(), "the resource is not registered");
    _LIBCPP_ASSERT_INTERNAL(__it->second.__count > 0, "the resouce is not active");
    return __it;
  }
};

_LIBCPP_EXPORTED_FROM_ABI _LIBCPP_AVAILABILITY_SHARED_RESOURCE mutex&
__shared_resource_inc_reference(const void* __ptr) {
  return __shared_resource::__instance().__inc_reference(__ptr);
}

_LIBCPP_EXPORTED_FROM_ABI _LIBCPP_AVAILABILITY_SHARED_RESOURCE void __shared_resource_dec_reference(const void* __ptr) {
  __shared_resource::__instance().__dec_reference(__ptr);
}

[[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI _LIBCPP_AVAILABILITY_SHARED_RESOURCE lock_guard<mutex>
__shared_resource_get_lock(const void* __ptr) {
  return __shared_resource::__instance().__get_lock(__ptr);
}

_LIBCPP_END_NAMESPACE_STD
