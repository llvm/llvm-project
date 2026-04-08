//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains declarations and utilities for liboffload mocking in
/// libsycl unit tests.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_UNITTESTS_MOCK_HELPERS_HPP
#define _LIBSYCL_UNITTESTS_MOCK_HELPERS_HPP

#include <OffloadAPI.h>

#include <gmock/gmock.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace mock {

struct ol_dummy_handle_t {
  ol_dummy_handle_t(size_t DataSize = 0)
      : MStorage(DataSize), MSize(DataSize) {}
  ol_dummy_handle_t(unsigned char *Data, size_t Size)
      : MStorage(Size), MSize(Size) {
    std::memcpy(MStorage.data(), Data, Size);
  }
  std::atomic<size_t> MRefCounter = 1;
  std::vector<unsigned char> MStorage;
  size_t MSize;

  template <typename T> T getDataAs() {
    assert(MStorage.size() >= sizeof(T));
    return *reinterpret_cast<T *>(MStorage.data());
  }

  template <typename T> T setDataAs(T Val) {
    assert(MStorage.size() >= sizeof(T));
    return *reinterpret_cast<T *>(MStorage.data()) = Val;
  }
};

using dummy_handle_t = ol_dummy_handle_t *;

template <class T> inline T createDummyHandle(size_t Size = 0) {
  dummy_handle_t DummyHandlePtr = new ol_dummy_handle_t(Size);
  return reinterpret_cast<T>(DummyHandlePtr);
}

template <class T>
inline T createDummyHandleWithData(unsigned char *Data, size_t Size) {
  auto DummyHandlePtr = new ol_dummy_handle_t(Data, Size);
  return reinterpret_cast<T>(DummyHandlePtr);
}

template <class T> inline void releaseDummyHandle(T Handle) {
  auto DummyHandlePtr = reinterpret_cast<dummy_handle_t>(Handle);
  delete DummyHandlePtr;
}

class MockLiboffload {
public:
  MockLiboffload() { initDefault(); }

  MOCK_METHOD(ol_result_t, olInit, (const ol_init_args_t *));
  MOCK_METHOD(ol_result_t, olShutDown, ());
  MOCK_METHOD(ol_result_t, olGetPlatformInfoSize,
              (ol_platform_handle_t, ol_platform_info_t, size_t *));
  MOCK_METHOD(ol_result_t, olGetPlatformInfo,
              (ol_platform_handle_t Platform, ol_platform_info_t PropName,
               size_t PropSize, void *PropValue));
  MOCK_METHOD(ol_result_t, olGetDeviceInfo,
              (ol_device_handle_t Device, ol_device_info_t PropName,
               size_t PropSize, void *PropValue));
  MOCK_METHOD(ol_result_t, olGetDeviceInfoSize,
              (ol_device_handle_t Device, ol_device_info_t PropName,
               size_t *PropSizeRet));
  MOCK_METHOD(ol_result_t, olIterateDevices,
              (ol_device_iterate_cb_t Callback, void *UserData));
  MOCK_METHOD(ol_result_t, olDestroyProgram, (ol_program_handle_t Program));
  MOCK_METHOD(ol_result_t, olCreateQueue,
              (ol_device_handle_t Device, ol_queue_handle_t *Queue));
  MOCK_METHOD(ol_result_t, olDestroyQueue, (ol_queue_handle_t Queue));
  MOCK_METHOD(ol_result_t, olSyncQueue, (ol_queue_handle_t Queue));
  MOCK_METHOD(ol_result_t, olDestroyEvent, (ol_event_handle_t Event));
  MOCK_METHOD(ol_result_t, olCreateProgram,
              (ol_device_handle_t Device, const void *ProgData,
               size_t ProgDataSize, ol_program_handle_t *Program));

  MOCK_METHOD(ol_result_t, olGetSymbol,
              (ol_program_handle_t Program, const char *Name,
               ol_symbol_kind_t Kind, ol_symbol_handle_t *Symbol));
  MOCK_METHOD(ol_result_t, olIsValidBinary,
              (ol_device_handle_t Device, const void *ProgData,
               size_t ProgDataSize, bool *Valid));
  MOCK_METHOD(ol_result_t, olWaitEvents,
              (ol_queue_handle_t Queue, ol_event_handle_t *Events,
               size_t NumEvents));
  MOCK_METHOD(ol_result_t, olCreateEvent,
              (ol_queue_handle_t Queue, ol_event_handle_t *Event));

  void initDefault();

  ol_result_t makeEmptyStrError(ol_errc_t Code) {
    auto [Iterator, Flag] =
        Errors.emplace(std::make_pair(Code, ol_error_struct_t{Code, ""}));
    return &Iterator->second;
  }

private:
  std::unordered_map<ol_errc_t, ol_error_struct_t> Errors;
  ol_platform_handle_t DefaultPlatform;
  ol_device_handle_t DefaultDevice{};
};

#ifndef _LIB_EXPORT
#  ifdef _WIN32
#    define _LIB_EXPORT __declspec(dllexport)
#  else // _WIN32
#    define _LIB_EXPORT __attribute__((visibility("default")))
#  endif // _WIN32
#endif   // _LIB_EXPORT

_LIB_EXPORT MockLiboffload &getMockLiboffload();

class MockWrapper {
public:
  MockWrapper() : Mock(getMockLiboffload()) {}
  ~MockWrapper() {
    ::testing::Mock::VerifyAndClearExpectations(&Mock); // move to common
  }
  MockLiboffload &get() { return Mock; };

private:
  MockLiboffload &Mock;
};

} // namespace mock

#endif // _LIBSYCL_UNITTESTS_MOCK_HELPERS_HPP
