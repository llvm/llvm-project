//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>

#include "helpers.hpp"

ol_result_t olInit(const ol_init_args_t *InitArgs) {
  return unittest::getMockLiboffload().olInit(InitArgs);
}

ol_result_t olShutDown() { return unittest::getMockLiboffload().olShutDown(); }

ol_result_t olGetPlatformInfoSize(ol_platform_handle_t Platform,
                                  ol_platform_info_t PropName,
                                  size_t *PropSizeRet) {
  return unittest::getMockLiboffload().olGetPlatformInfoSize(Platform, PropName,
                                                             PropSizeRet);
}

OL_APIEXPORT ol_result_t OL_APICALL
olGetPlatformInfo(ol_platform_handle_t Platform, ol_platform_info_t PropName,
                  size_t PropSize, void *PropValue) {
  return unittest::getMockLiboffload().olGetPlatformInfo(Platform, PropName,
                                                         PropSize, PropValue);
}

ol_result_t olGetDeviceInfo(ol_device_handle_t Device,
                            ol_device_info_t PropName, size_t PropSize,
                            void *PropValue) {
  return unittest::getMockLiboffload().olGetDeviceInfo(Device, PropName,
                                                       PropSize, PropValue);
}

ol_result_t olGetDeviceInfoSize(ol_device_handle_t Device,
                                ol_device_info_t PropName,
                                size_t *PropSizeRet) {
  return unittest::getMockLiboffload().olGetDeviceInfoSize(Device, PropName,
                                                           PropSizeRet);
}

ol_result_t olIterateDevices(ol_device_iterate_cb_t Callback, void *UserData) {
  return unittest::getMockLiboffload().olIterateDevices(Callback, UserData);
}

ol_result_t olDestroyProgram(ol_program_handle_t Program) {
  return unittest::getMockLiboffload().olDestroyProgram(Program);
}

ol_result_t olCreateQueue(ol_device_handle_t Device, ol_queue_handle_t *Queue) {
  return unittest::getMockLiboffload().olCreateQueue(Device, Queue);
}

ol_result_t olDestroyQueue(ol_queue_handle_t Queue) {
  return unittest::getMockLiboffload().olDestroyQueue(Queue);
}

ol_result_t olSyncQueue(ol_queue_handle_t Queue) {
  return unittest::getMockLiboffload().olSyncQueue(Queue);
}

ol_result_t olCreateProgram(
    ol_device_handle_t Device,
    const void *ProgData,
    size_t ProgDataSize,
    ol_program_handle_t *Program)
    {
       return unittest::getMockLiboffload().olCreateProgram(Device, ProgData, ProgDataSize, Program);
    }

    ol_result_t olGetSymbol(ol_program_handle_t Program, const char *Name,
                            ol_symbol_kind_t Kind, ol_symbol_handle_t *Symbol) {
      return unittest::getMockLiboffload().olGetSymbol(Program, Name, Kind,
                                                       Symbol);
    }

    ol_result_t olIsValidBinary(ol_device_handle_t Device, const void *ProgData,
                                size_t ProgDataSize, bool *Valid) {
      return unittest::getMockLiboffload().olIsValidBinary(Device, ProgData,
                                                           ProgDataSize, Valid);
    }

    ol_result_t olWaitEvents(ol_queue_handle_t Queue, ol_event_handle_t *Events,
                             size_t NumEvents) {
      return unittest::getMockLiboffload().olWaitEvents(Queue, Events,
                                                        NumEvents);
    }

    ol_result_t
    olLaunchKernel(ol_queue_handle_t Queue, ol_device_handle_t Device,
                   ol_symbol_handle_t Kernel, const void *ArgumentsData,
                   size_t ArgumentsSize,
                   const ol_kernel_launch_size_args_t *LaunchSizeArgs) {
      return unittest::getMockLiboffload().olLaunchKernel(
          Queue, Device, Kernel, ArgumentsData, ArgumentsSize, LaunchSizeArgs);
    }

    ol_result_t olCreateEvent(ol_queue_handle_t Queue,
                              ol_event_handle_t *Event) {
      return unittest::getMockLiboffload().olCreateEvent(Queue, Event);
    }

    ol_result_t olDestroyEvent(ol_event_handle_t Event) {
      return unittest::getMockLiboffload().olDestroyEvent(Event);
    }
