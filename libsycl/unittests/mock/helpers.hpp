#pragma once

#include <OffloadAPI.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace mock {

struct dummy_handle_t_ {
  dummy_handle_t_(size_t DataSize = 0) : MStorage(DataSize), MSize(DataSize) {}
  dummy_handle_t_(unsigned char *Data, size_t Size)
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

using dummy_handle_t = dummy_handle_t_ *;

template <class T> inline T createDummyHandle(size_t Size = 0) {
  dummy_handle_t DummyHandlePtr = new dummy_handle_t_(Size);
  return reinterpret_cast<T>(DummyHandlePtr);
}

template <class T>
inline T createDummyHandleWithData(unsigned char *Data, size_t Size) {
  auto DummyHandlePtr = new dummy_handle_t_(Data, Size);
  return reinterpret_cast<T>(DummyHandlePtr);
}

template <class T> inline void releaseDummyHandle(T Handle) {
  auto DummyHandlePtr = reinterpret_cast<dummy_handle_t>(Handle);
  delete DummyHandlePtr;
}
} // namespace mock

namespace unittest {

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

} // namespace unittest

class MockWrapper {
public:
  MockWrapper() : Mock(unittest::getMockLiboffload()) {}
  ~MockWrapper() {
    ::testing::Mock::VerifyAndClearExpectations(&Mock); // move to common
  }
  unittest::MockLiboffload &get() { return Mock; };

private:
  unittest::MockLiboffload &Mock;
};
