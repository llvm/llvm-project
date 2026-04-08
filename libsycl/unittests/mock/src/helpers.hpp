#pragma once

#include <OffloadAPI.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// This is the callback function we accept to override or instrument
// entry-points. pParams is expected to be a pointer to the appropriate params_t
// struct for the given entry point.
typedef ol_result_t (*ol_mock_callback_t)(void *pParams);

namespace mock {

struct dummy_handle_t_ {
  dummy_handle_t_(size_t DataSize = 0)
      : MStorage(DataSize), MData(MStorage.data()), MSize(DataSize) {}
  dummy_handle_t_(unsigned char *Data, size_t Size)
      : MData(Data), MSize(Size) {}
  std::atomic<size_t> MRefCounter = 1;
  std::vector<unsigned char> MStorage;
  unsigned char *MData = nullptr;
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

struct Callbacks {
  void setCallback(std::string name, ol_mock_callback_t callback) {
    replaceCallbacks[name] = callback;
  }

  ol_mock_callback_t getCallback(std::string name) const {
    auto callback = replaceCallbacks.find(name);

    if (callback != replaceCallbacks.end()) {
      return callback->second;
    }
    return nullptr;
  }

  void resetCallbacks() { replaceCallbacks.clear(); }
  ol_error_struct_t *
  getErrorUnimplementedFunction(const std::string &FunctionName) {
    if (auto ErrorIt = errors.find(FunctionName); ErrorIt != errors.end())
      return &ErrorIt->second.second;
    auto [Iterator, Flag] = errors.insert(
        {FunctionName,
         {FunctionName + " is not implemented in mock OL library, add callback "
                         "via setCAllback method.",
          {}}});
    assert(Flag);
    auto &[MessageStr, ErrorStruct] = Iterator->second;
    ErrorStruct = {OL_ERRC_UNIMPLEMENTED, MessageStr.c_str()};
    return &ErrorStruct;
  }

private:
  std::unordered_map<std::string, ol_mock_callback_t> replaceCallbacks;
  std::unordered_map<std::string, std::pair<std::string, ol_error_struct_t>>
      errors;
};

#ifndef _LIB_EXPORT
#  ifdef _WIN32
#    define _LIB_EXPORT __declspec(dllexport)
#  else // _WIN32
#    define _LIB_EXPORT __attribute__((visibility("default")))
#  endif // _WIN32
#endif   // _LIB_EXPORT

_LIB_EXPORT Callbacks &getCallbacks();
_LIB_EXPORT ol_error_struct_t *
getErrorUnimplementedFunction(const std::string &FunctionName);

} // namespace mock
