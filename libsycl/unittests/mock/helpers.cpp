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
        EXPECT_TRUE(!InitArgs || (*InitArgs == DefaultArgs));

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
        EXPECT_NE(Platform, nullptr);
        EXPECT_NE(PropSizeRet, nullptr);

        EXPECT_EQ(PropName, OL_PLATFORM_INFO_BACKEND);
        *PropSizeRet = sizeof(ol_platform_backend_t);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olGetPlatformInfo)
      .WillByDefault([this](ol_platform_handle_t Platform,
                            ol_platform_info_t PropName, size_t PropSize,
                            void *PropValue) -> ol_result_t {
        EXPECT_NE(Platform, nullptr);
        EXPECT_EQ(PropName, OL_PLATFORM_INFO_BACKEND);
        EXPECT_EQ(PropSize, sizeof(ol_platform_backend_t));
        EXPECT_NE(PropValue, nullptr);

        assignAs<ol_platform_backend_t>(PropValue,
                                        Platform == HostPlatform
                                            ? OL_PLATFORM_BACKEND_HOST
                                            : OL_PLATFORM_BACKEND_LEVEL_ZERO);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olGetDeviceInfo)
      .WillByDefault([this](ol_device_handle_t Device,
                            ol_device_info_t PropName, size_t PropSize,
                            void *PropValue) -> ol_result_t {
        EXPECT_NE(Device, nullptr);
        EXPECT_GT(PropSize, 0);
        EXPECT_NE(PropValue, nullptr);

        switch (PropName) {
        case OL_DEVICE_INFO_PLATFORM: {
          EXPECT_EQ(PropSize, sizeof(ol_platform_handle_t));
          assignAs<ol_platform_handle_t>(
              PropValue, reinterpret_cast<mock::dummy_handle_t>(Device)
                             ->getDataAs<ol_platform_handle_t>());
          return OL_SUCCESS;
        }
        case OL_DEVICE_INFO_TYPE: {
          EXPECT_EQ(PropSize, sizeof(ol_device_type_t));
          assignAs<ol_device_type_t>(PropValue, OL_DEVICE_TYPE_GPU);
          return OL_SUCCESS;
        }
        case OL_DEVICE_INFO_CONTEXT_GROUP_INDEX: {
          EXPECT_EQ(PropSize, sizeof(uint32_t));
          assignAs<uint32_t>(PropValue, 0);
          return OL_SUCCESS;
        }
        default:
          ADD_FAILURE();
          return makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);
        }
      });

  ON_CALL(*this, olGetDeviceInfoSize)
      .WillByDefault([this](ol_device_handle_t Device,
                            ol_device_info_t PropName,
                            size_t *PropSizeRet) -> ol_result_t {
        EXPECT_NE(Device, nullptr);
        EXPECT_NE(PropSizeRet, nullptr);
        switch (PropName) {
        case OL_DEVICE_INFO_PLATFORM: {
          *PropSizeRet = sizeof(ol_platform_handle_t);
          return OL_SUCCESS;
        }
        case OL_DEVICE_INFO_TYPE: {
          *PropSizeRet = sizeof(ol_device_type_t);
          return OL_SUCCESS;
        }
        case OL_DEVICE_INFO_CONTEXT_GROUP_INDEX: {
          *PropSizeRet = sizeof(uint32_t);
          return OL_SUCCESS;
        }
        default:
          ADD_FAILURE();
          return makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);
        }
      });

  ON_CALL(*this, olIterateDevices)
      .WillByDefault([this](ol_device_iterate_cb_t Callback,
                            void *UserData) -> ol_result_t {
        EXPECT_NE(Callback, nullptr);

        assert(DefaultDevice);
        std::ignore = Callback(DefaultDevice, UserData);
        assert(HostDevice);
        std::ignore = Callback(HostDevice, UserData);

        return OL_SUCCESS;
      });

  ON_CALL(*this, olDestroyProgram)
      .WillByDefault([this](ol_program_handle_t Program) -> ol_result_t {
        EXPECT_NE(Program, nullptr);
        mock::releaseDummyHandle(Program);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olCreateContext)
      .WillByDefault([this](size_t NumDevices, ol_device_handle_t *Devices,
                            ol_context_handle_t *Context) -> ol_result_t {
        EXPECT_GT(NumDevices, 0);
        EXPECT_NE(Devices, nullptr);
        EXPECT_NE(Context, nullptr);

        for (size_t I = 0; I < NumDevices; ++I) {
          EXPECT_NE(Devices[I], nullptr);
        }

        // Preserve the first device in payload for tests that may need to
        // inspect what device set the context was created from.
        *Context = mock::createDummyHandleWithData<ol_context_handle_t>(
            reinterpret_cast<unsigned char *>(&Devices[0]), sizeof(Devices[0]));
        return OL_SUCCESS;
      });

  ON_CALL(*this, olDestroyContext)
      .WillByDefault([this](ol_context_handle_t Context) -> ol_result_t {
        EXPECT_NE(Context, nullptr);
        mock::releaseDummyHandle(Context);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olCreateProgram)
      .WillByDefault([this](ol_device_handle_t Device, const void *ProgData,
                            size_t ProgDataSize,
                            ol_program_handle_t *Program) -> ol_result_t {
        EXPECT_NE(Device, nullptr);
        EXPECT_NE(ProgData, nullptr);
        EXPECT_GT(ProgDataSize, 0);
        EXPECT_NE(Program, nullptr);
        *Program = mock::createDummyHandleWithData<ol_program_handle_t>(
            reinterpret_cast<unsigned char *>(&Device), sizeof(Device));
        return OL_SUCCESS;
      });

  ON_CALL(*this, olIsValidBinary)
      .WillByDefault([this](ol_device_handle_t Device, const void *ProgData,
                            size_t ProgDataSize, bool *Valid) -> ol_result_t {
        EXPECT_NE(Device, nullptr);
        EXPECT_NE(ProgData, nullptr);
        EXPECT_GT(ProgDataSize, 0);
        EXPECT_NE(Valid, nullptr);
        *Valid = true;
        return OL_SUCCESS;
      });

  ON_CALL(*this, olGetSymbol)
      .WillByDefault([this](ol_program_handle_t Program, const char *Name,
                            ol_symbol_kind_t Kind,
                            ol_symbol_handle_t *Symbol) -> ol_result_t {
        EXPECT_NE(Program, nullptr);
        EXPECT_NE(Name, nullptr);
        std::ignore = Kind;
        EXPECT_NE(Symbol, nullptr);
        *Symbol = mock::createDummyHandleWithData<ol_symbol_handle_t>(
            reinterpret_cast<unsigned char *>(&Program), sizeof(Program));
        return OL_SUCCESS;
      });

  ON_CALL(*this, olCreateQueue)
      .WillByDefault([this](ol_context_handle_t Context,
                            ol_device_handle_t Device,
                            ol_queue_handle_t *Queue) -> ol_result_t {
        std::ignore = Context;
        EXPECT_NE(Device, nullptr);
        EXPECT_NE(Queue, nullptr);
        // Attach device as data to check what device queue belongs to if needed
        *Queue = mock::createDummyHandleWithData<ol_queue_handle_t>(
            reinterpret_cast<unsigned char *>(&Device), sizeof(Device));
        return OL_SUCCESS;
      });

  ON_CALL(*this, olDestroyQueue)
      .WillByDefault([this](ol_queue_handle_t Queue) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        mock::releaseDummyHandle(Queue);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olSyncQueue)
      .WillByDefault([this](ol_queue_handle_t Queue) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        std::ignore = Queue;
        return OL_SUCCESS;
      });

  ON_CALL(*this, olWaitEvents)
      .WillByDefault([this](ol_queue_handle_t Queue, ol_event_handle_t *Events,
                            size_t NumEvents) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        EXPECT_NE(Events, nullptr);
        for (size_t I = 0; I < NumEvents; ++I) {
          EXPECT_NE(Events[I], nullptr);
        }
        return OL_SUCCESS;
      });

  ON_CALL(*this, olCreateEvent)
      .WillByDefault([this](ol_queue_handle_t Queue, ol_event_flags_t Flags,
                            ol_event_handle_t *Event) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        std::ignore = Flags;
        EXPECT_NE(Event, nullptr);
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
        EXPECT_NE(Queue, nullptr);
        EXPECT_NE(Device, nullptr);
        EXPECT_NE(Kernel, nullptr);
        EXPECT_NE(LaunchSizeArgs, nullptr);
        std::ignore = Properties;
        EXPECT_TRUE(NumArgs == 0 || ArgPtrs != nullptr);
        EXPECT_EQ(!ArgPtrs, !ArgSizes);
        for (size_t I = 0; I < NumArgs; ++I) {
          EXPECT_TRUE(ArgSizes[I] == 0 || ArgPtrs[I] != nullptr);
        }
        return OL_SUCCESS;
      });

  ON_CALL(*this, olSyncEvent)
      .WillByDefault([this](ol_event_handle_t Event) -> ol_result_t {
        EXPECT_NE(Event, nullptr);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olMemcpy)
      .WillByDefault([this](ol_queue_handle_t Queue, void *DstPtr,
                            ol_device_handle_t DstDevice, const void *SrcPtr,
                            ol_device_handle_t SrcDevice,
                            size_t Size) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        EXPECT_NE(DstPtr, nullptr);
        EXPECT_NE(DstDevice, nullptr);
        EXPECT_NE(SrcPtr, nullptr);
        EXPECT_NE(SrcDevice, nullptr);
        return OL_SUCCESS;
      });
  ON_CALL(*this, olMemPrefetch)
      .WillByDefault([this](ol_queue_handle_t Queue, size_t Count,
                            const void **Mems, const size_t *Sizes,
                            ol_mem_migration_flags_t Flags) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        EXPECT_EQ(Count, 1);
        EXPECT_NE(Mems, nullptr);
        EXPECT_NE(*Mems, nullptr);
        EXPECT_NE(Sizes, nullptr);
        EXPECT_GT(*Sizes, 0);
        EXPECT_EQ(Flags, OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE);
        return OL_SUCCESS;
      });
  ON_CALL(*this, olGetMemInfo)
      .WillByDefault([this](const void *Ptr, ol_mem_info_t PropName,
                            size_t PropSize, void *PropValue) -> ol_result_t {
        EXPECT_NE(Ptr, nullptr);
        // Other properties are not used by the runtime yet
        EXPECT_EQ(PropName, OL_MEM_INFO_DEVICE);
        EXPECT_EQ(PropSize, sizeof(ol_device_handle_t));
        assignAs(PropValue, DefaultDevice);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olDestroyEvent)
      .WillByDefault([this](ol_event_handle_t Event) -> ol_result_t {
        EXPECT_NE(Event, nullptr);
        mock::releaseDummyHandle(Event);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olMemAlloc)
      .WillByDefault([this](ol_device_handle_t Device, ol_alloc_type_t Type,
                            size_t Size, void **AllocationOut) -> ol_result_t {
        EXPECT_NE(Device, nullptr);
        EXPECT_NE(Type, OL_ALLOC_TYPE_HOST);
        EXPECT_GT(Size, 0);
        EXPECT_NE(AllocationOut, nullptr);
        *AllocationOut = mock::createDummyHandle<void *>(Size);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olMemAllocHost)
      .WillByDefault([this](ol_device_handle_t Device, size_t Size,
                            void **AllocationOut) -> ol_result_t {
        EXPECT_NE(Device, nullptr);
        EXPECT_GT(Size, 0);
        EXPECT_NE(AllocationOut, nullptr);
        *AllocationOut = mock::createDummyHandle<void *>(Size);
        return OL_SUCCESS;
      });

  ON_CALL(*this, olMemFree).WillByDefault([this](void *Address) -> ol_result_t {
    EXPECT_NE(Address, nullptr);
    mock::releaseDummyHandle(Address);
    return OL_SUCCESS;
  });
}
