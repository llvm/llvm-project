//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helpers.hpp"

namespace mock {

_LIB_EXPORT MockLiboffload &getMockLiboffload() {
  static MockLiboffload Mock;
  return Mock;
}

} // namespace mock

static bool operator==(const ol_init_args_t &lhs, const ol_init_args_t &rhs) {
  return (lhs.Size == rhs.Size) && (lhs.NumPlatforms == rhs.NumPlatforms) &&
         std::memcmp(lhs.Platforms, rhs.Platforms,
                     lhs.NumPlatforms * sizeof(ol_platform_backend_t)) == 0;
}

static bool operator!=(const ol_init_args_t &lhs, const ol_init_args_t &rhs) {
  return !(lhs == rhs);
}

template <typename T> void assignAs(void *PropValue, T NewValue) {
  *(static_cast<T *>(PropValue)) = NewValue;
}

void mock::MockLiboffload::initDefault() {
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
            reinterpret_cast<unsigned char *>(&DefaultPlatform),
            sizeof(DefaultPlatform));
        HostPlatform = mock::createDummyHandle<ol_platform_handle_t>();
        HostDevice = mock::createDummyHandleWithData<ol_device_handle_t>(
            reinterpret_cast<unsigned char *>(&HostPlatform),
            sizeof(HostPlatform));

        return OL_SUCCESS;
      });

  ON_CALL(*this, olShutDown).WillByDefault([this]() -> ol_result_t {
    assert(DefaultDevice);

    mock::releaseDummyHandle(DefaultPlatform);
    mock::releaseDummyHandle(DefaultDevice);
    mock::releaseDummyHandle(HostPlatform);
    mock::releaseDummyHandle(HostDevice);

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
                                          Platform == HostPlatform
                                              ? OL_PLATFORM_BACKEND_HOST
                                              : OL_PLATFORM_BACKEND_LEVEL_ZERO);
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
        std::ignore = Callback(DefaultDevice, UserData);
        assert(HostDevice);
        std::ignore = Callback(HostDevice, UserData);

        return OL_SUCCESS;
      });

  ON_CALL(*this, olDestroyProgram)
      .WillByDefault([this](ol_program_handle_t Program) -> ol_result_t {
        if (!Program)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        mock::releaseDummyHandle(Program);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olCreateProgram)
      .WillByDefault([this](ol_device_handle_t Device, const void *ProgData,
                            size_t ProgDataSize,
                            ol_program_handle_t *Program) -> ol_result_t {
        if (!Device)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!ProgData || !Program || !ProgDataSize)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);

        *Program = mock::createDummyHandleWithData<ol_program_handle_t>(
            reinterpret_cast<unsigned char *>(&Device), sizeof(Device));
        return OL_SUCCESS;
      });

  ON_CALL(*this, olIsValidBinary)
      .WillByDefault([this](ol_device_handle_t Device, const void *ProgData,
                            size_t ProgDataSize, bool *Valid) -> ol_result_t {
        if (!Device)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!ProgData || !Valid || !ProgDataSize)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);
        *Valid = true;
        return OL_SUCCESS;
      });

  ON_CALL(*this, olGetSymbol)
      .WillByDefault([this](ol_program_handle_t Program, const char *Name,
                            ol_symbol_kind_t Kind,
                            ol_symbol_handle_t *Symbol) -> ol_result_t {
        if (!Program)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!Name || !Symbol)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);
        std::ignore = Kind;

        *Symbol = mock::createDummyHandleWithData<ol_symbol_handle_t>(
            reinterpret_cast<unsigned char *>(&Program), sizeof(Program));
        return OL_SUCCESS;
      });

  ON_CALL(*this, olCreateQueue)
      .WillByDefault([this](ol_device_handle_t Device,
                            ol_queue_handle_t *Queue) -> ol_result_t {
        if (!Device)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!Queue)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);
        // Attach device as data to check what device queue belongs to if needed
        *Queue = mock::createDummyHandleWithData<ol_queue_handle_t>(
            reinterpret_cast<unsigned char *>(&Device), sizeof(Device));
        return OL_SUCCESS;
      });

  ON_CALL(*this, olDestroyQueue)
      .WillByDefault([this](ol_queue_handle_t Queue) -> ol_result_t {
        if (!Queue)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        mock::releaseDummyHandle(Queue);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olSyncQueue)
      .WillByDefault([this](ol_queue_handle_t Queue) -> ol_result_t {
        if (!Queue)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        std::ignore = Queue;
        return OL_SUCCESS;
      });

  ON_CALL(*this, olWaitEvents)
      .WillByDefault([this](ol_queue_handle_t Queue, ol_event_handle_t *Events,
                            size_t NumEvents) -> ol_result_t {
        if (!Queue)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!Events)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);
        for (size_t I = 0; I < NumEvents; ++I) {
          if (!Events[I])
            return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        }
        return OL_SUCCESS;
      });

  ON_CALL(*this, olCreateEvent)
      .WillByDefault([this](ol_queue_handle_t Queue, ol_event_flags_t Flags,
                            ol_event_handle_t *Event) -> ol_result_t {
        if (!Queue)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!Event)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);
        std::ignore = Flags;
        *Event = mock::createDummyHandleWithData<ol_event_handle_t>(
            reinterpret_cast<unsigned char *>(&Queue), sizeof(Queue));
        return OL_SUCCESS;
      });

  ON_CALL(*this, olLaunchKernel)
      .WillByDefault([this](ol_queue_handle_t Queue, ol_device_handle_t Device,
                            ol_symbol_handle_t Kernel,
                            const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                            const ol_kernel_launch_prop_t *Properties,
                            size_t NumArgs, void **ArgPtrs,
                            const size_t *ArgSizes) -> ol_result_t {
        if (!Device || !Kernel || !Queue)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!LaunchSizeArgs)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);
        std::ignore = Properties;
        if ((ArgPtrs == nullptr) != (ArgSizes == nullptr))
          return makeEmptyStrError(OL_ERRC_INVALID_ARGUMENT);
        if (NumArgs > 0 && (!ArgPtrs || !ArgSizes))
          return makeEmptyStrError(OL_ERRC_INVALID_ARGUMENT);
        for (size_t I = 0; I < NumArgs; ++I) {
          if (ArgSizes[I] > 0 && !ArgPtrs[I])
            return makeEmptyStrError(OL_ERRC_INVALID_ARGUMENT);
        }
        return OL_SUCCESS;
      });

  ON_CALL(*this, olSyncEvent)
      .WillByDefault([this](ol_event_handle_t Event) -> ol_result_t {
        if (!Event)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olMemcpy)
      .WillByDefault([this](ol_queue_handle_t Queue, void *DstPtr,
                            ol_device_handle_t DstDevice, const void *SrcPtr,
                            ol_device_handle_t SrcDevice,
                            size_t Size) -> ol_result_t {
        if (!Queue || !DstDevice || !DstDevice)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        if (!DstPtr || !SrcPtr)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olGetMemInfo)
      .WillByDefault([this](const void *Ptr, ol_mem_info_t PropName,
                            size_t PropSize, void *PropValue) -> ol_result_t {
        if (!Ptr)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_POINTER);
        // Other properties are not used by the runtime yet
        EXPECT_EQ(PropName, OL_MEM_INFO_DEVICE);
        if (PropSize < sizeof(ol_device_handle_t))
          return makeEmptyStrError(OL_ERRC_INVALID_SIZE);
        assignAs(PropValue, DefaultDevice);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olDestroyEvent)
      .WillByDefault([this](ol_event_handle_t Event) -> ol_result_t {
        if (!Event)
          return makeEmptyStrError(OL_ERRC_INVALID_NULL_HANDLE);
        mock::releaseDummyHandle(Event);
        return OL_SUCCESS;
      });
}
