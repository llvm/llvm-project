#include <cstring>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>

#include "helpers.hpp"

std::unordered_map<ol_errc_t, ol_error_struct_t> Errors;
ol_platform_handle_t DefaultPlatform;
ol_device_handle_t DefaultDevice{};

ol_result_t makeEmptyStrError(ol_errc_t Code) {
  auto [Iterator, Flag] =
      Errors.emplace(std::make_pair(Code, ol_error_struct_t{Code, ""}));
  return &Iterator->second;
}

// C++20 std::source_location::function_name
ol_result_t olCreateEvent(ol_queue_handle_t Queue, ol_event_handle_t *Event) {
  return unittest::OffloadMock::callCallback<ol_create_event_params_t>(
      __func__, Queue, Event);
}

bool operator==(const ol_init_args_t &lhs, const ol_init_args_t &rhs) {
  return (lhs.Size == rhs.Size) && (lhs.NumPlatforms == rhs.NumPlatforms) &&
         std::memcmp(lhs.Platforms, rhs.Platforms,
                     lhs.NumPlatforms * sizeof(ol_platform_backend_t)) == 0;
}

bool operator!=(const ol_init_args_t &lhs, const ol_init_args_t &rhs) {
  return !(lhs == rhs);
}

ol_result_t olInit(const ol_init_args_t *InitArgs) {
  // TODO: complicated cases with non-default settings are not covered.
  const ol_init_args_t DefaultArgs = OL_INIT_ARGS_INIT;
  if (InitArgs && (*InitArgs != DefaultArgs))
    return makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);

  assert(!DefaultDevice);
  DefaultPlatform = mock::createDummyHandle<ol_platform_handle_t>();
  DefaultDevice = mock::createDummyHandleWithData<ol_device_handle_t>(
      reinterpret_cast<unsigned char *>(DefaultPlatform),
      sizeof(DefaultPlatform));

  return OL_SUCCESS;
}

ol_result_t olShutDown() {
  assert(DefaultDevice);

  // release platform.
  mock::releaseDummyHandle(DefaultPlatform);
  // release device.
  mock::releaseDummyHandle(DefaultDevice);

  return OL_SUCCESS;
}

ol_result_t olGetPlatformInfoSize(ol_platform_handle_t Platform,
                                  ol_platform_info_t PropName,
                                  size_t *PropSizeRet) {
  if (!Platform)
    return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
  if (!PropSizeRet)
    return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);

  if (PropName == OL_PLATFORM_INFO_BACKEND) {
    *PropSizeRet = sizeof(ol_platform_backend_t);
    return OL_SUCCESS;
  }

  return makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);
}

template <typename T> void assignAs(void *PropValue, T NewValue) {
  *(static_cast<T *>(PropValue)) = NewValue;
}

OL_APIEXPORT ol_result_t OL_APICALL
olGetPlatformInfo(ol_platform_handle_t Platform, ol_platform_info_t PropName,
                  size_t PropSize, void *PropValue) {
  if (!Platform)
    return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
  if (!PropSize)
    return makeEmptyStrError(OL_ERRC_INVALID_SIZE);
  if (!PropValue)
    return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);

  if (PropName == OL_PLATFORM_INFO_BACKEND) {
    if (PropSize != sizeof(ol_platform_backend_t))
      return makeEmptyStrError(OL_ERRC_INVALID_SIZE);
    assignAs<ol_platform_backend_t>(PropValue, OL_PLATFORM_BACKEND_LEVEL_ZERO);
    return OL_SUCCESS;
  }

  return makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);
}

ol_result_t olGetDeviceInfo(ol_device_handle_t Device,
                            ol_device_info_t PropName, size_t PropSize,
                            void *PropValue) {
  if (!Device)
    return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
  if (!PropSize)
    return makeEmptyStrError(OL_ERRC_INVALID_SIZE);
  if (!PropValue)
    return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);

  switch (PropName) {
  case OL_DEVICE_INFO_PLATFORM: {
    if (PropSize != sizeof(ol_platform_handle_t))
      return makeEmptyStrError(OL_ERRC_INVALID_SIZE);
    assignAs<ol_platform_handle_t>(
        PropValue, reinterpret_cast<mock::dummy_handle_t>(Device)
                       ->getDataAs<ol_platform_handle_t>());
    return OL_SUCCESS;
  }
  case OL_DEVICE_INFO_TYPE: {
    if (PropSize != sizeof(ol_device_type_t))
      return makeEmptyStrError(OL_ERRC_INVALID_SIZE);
    assignAs<ol_device_type_t>(PropValue, OL_DEVICE_TYPE_GPU);
    return OL_SUCCESS;
  }
  default:
    return makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);
  }
}

ol_result_t olGetDeviceInfoSize(ol_device_handle_t Device,
                                ol_device_info_t PropName,
                                size_t *PropSizeRet) {
  if (!Device)
    return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
  if (!PropSizeRet)
    return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);
  switch (PropName) {
  case OL_DEVICE_INFO_PLATFORM: {
    *PropSizeRet = sizeof(ol_platform_handle_t);
    return OL_SUCCESS;
  }
  case OL_DEVICE_INFO_TYPE: {
    *PropSizeRet = sizeof(ol_device_type_t);
    return OL_SUCCESS;
  }
  default:
    return makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);
  }
}

ol_result_t olIterateDevices(
    // [in] User-provided function called for each available device
    ol_device_iterate_cb_t Callback,
    // [in][optional] Optional user data to pass to the callback
    void *UserData) {
  if (!Callback)
    return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);

  assert(DefaultDevice);
  [[maybe_unused]] bool Result = Callback(DefaultDevice, UserData);

  return OL_SUCCESS;
}
