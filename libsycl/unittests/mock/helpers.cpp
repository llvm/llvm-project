#include "helpers.hpp"

namespace unittest {

_LIB_EXPORT MockLiboffload &getMockLiboffload() {
  static MockLiboffload Mock;
  return Mock;
}

} // namespace unittest

bool inline operator==(const ol_init_args_t &lhs, const ol_init_args_t &rhs) {
  return (lhs.Size == rhs.Size) && (lhs.NumPlatforms == rhs.NumPlatforms) &&
         std::memcmp(lhs.Platforms, rhs.Platforms,
                     lhs.NumPlatforms * sizeof(ol_platform_backend_t)) == 0;
}

bool inline operator!=(const ol_init_args_t &lhs, const ol_init_args_t &rhs) {
  return !(lhs == rhs);
}

template <typename T> void assignAs(void *PropValue, T NewValue) {
  *(static_cast<T *>(PropValue)) = NewValue;
}

void unittest::MockLiboffload::initDefault() {
  // Disable gmock warning of uninteresting mock calls.
  ::testing::FLAGS_gmock_verbose = "error";
  ON_CALL(*this, olInit)
      .WillByDefault([this](const ol_init_args_t *InitArgs) -> ol_result_t {
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
      });

  ON_CALL(*this, olShutDown).WillByDefault([this]() -> ol_result_t {
    assert(DefaultDevice);

    // release platform.
    mock::releaseDummyHandle(DefaultPlatform);
    // release device.
    mock::releaseDummyHandle(DefaultDevice);

    return OL_SUCCESS;
  });

  ON_CALL(*this, olGetPlatformInfoSize)
      .WillByDefault([this](ol_platform_handle_t Platform,
                            ol_platform_info_t PropName,
                            size_t *PropSizeRet) -> ol_result_t {
        if (!Platform)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!PropSizeRet)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);

        if (PropName == OL_PLATFORM_INFO_BACKEND) {
          *PropSizeRet = sizeof(ol_platform_backend_t);
          return OL_SUCCESS;
        }

        return makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);
      });

  ON_CALL(*this, olGetPlatformInfo)
      .WillByDefault([this](ol_platform_handle_t Platform,
                            ol_platform_info_t PropName, size_t PropSize,
                            void *PropValue) -> ol_result_t {
        if (!Platform)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!PropSize)
          return makeEmptyStrError(OL_ERRC_INVALID_SIZE);
        if (!PropValue)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);

        if (PropName == OL_PLATFORM_INFO_BACKEND) {
          if (PropSize != sizeof(ol_platform_backend_t))
            return makeEmptyStrError(OL_ERRC_INVALID_SIZE);
          assignAs<ol_platform_backend_t>(PropValue,
                                          OL_PLATFORM_BACKEND_LEVEL_ZERO);
          return OL_SUCCESS;
        }

        return makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);
      });

  ON_CALL(*this, olGetDeviceInfo)
      .WillByDefault([this](ol_device_handle_t Device,
                            ol_device_info_t PropName, size_t PropSize,
                            void *PropValue) -> ol_result_t {
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
      });

  ON_CALL(*this, olGetDeviceInfoSize)
      .WillByDefault([this](ol_device_handle_t Device,
                            ol_device_info_t PropName,
                            size_t *PropSizeRet) -> ol_result_t {
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
      });

  ON_CALL(*this, olIterateDevices)
      .WillByDefault([this](ol_device_iterate_cb_t Callback,
                            void *UserData) -> ol_result_t {
        if (!Callback)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);

        assert(DefaultDevice);
        [[maybe_unused]] bool Result = Callback(DefaultDevice, UserData);

        return OL_SUCCESS;
      });

  ON_CALL(*this, olDestroyProgram)
      .WillByDefault([](ol_program_handle_t Program) -> ol_result_t {
        mock::releaseDummyHandle(Program);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olCreateQueue)
      .WillByDefault([](ol_device_handle_t Device,
                        ol_queue_handle_t *Queue) -> ol_result_t {
        assert(Device);
        // Attach device as data to check what device queue belongs to if needed
        *Queue = mock::createDummyHandleWithData<ol_queue_handle_t>(
            reinterpret_cast<unsigned char *>(Device), sizeof(Device));
        return OL_SUCCESS;
      });

  ON_CALL(*this, olDestroyQueue)
      .WillByDefault([](ol_queue_handle_t Queue) -> ol_result_t {
        mock::releaseDummyHandle(Queue);
        return OL_SUCCESS;
      });
}
