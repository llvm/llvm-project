//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helpers.hpp"

ol_result_t olInit(const ol_init_args_t *InitArgs) {
  return mock::getMockLiboffload().olInit(InitArgs);
}

ol_result_t olShutDown() { return mock::getMockLiboffload().olShutDown(); }

ol_result_t olGetPlatformInfoSize(ol_platform_handle_t Platform,
                                  ol_platform_info_t PropName,
                                  size_t *PropSizeRet) {
  return mock::getMockLiboffload().olGetPlatformInfoSize(Platform, PropName,
                                                         PropSizeRet);
}

ol_result_t olGetPlatformInfo(ol_platform_handle_t Platform,
                              ol_platform_info_t PropName, size_t PropSize,
                              void *PropValue) {
  return mock::getMockLiboffload().olGetPlatformInfo(Platform, PropName,
                                                     PropSize, PropValue);
}

ol_result_t olGetDeviceInfo(ol_device_handle_t Device,
                            ol_device_info_t PropName, size_t PropSize,
                            void *PropValue) {
  return mock::getMockLiboffload().olGetDeviceInfo(Device, PropName, PropSize,
                                                   PropValue);
}

ol_result_t olGetDeviceInfoSize(ol_device_handle_t Device,
                                ol_device_info_t PropName,
                                size_t *PropSizeRet) {
  return mock::getMockLiboffload().olGetDeviceInfoSize(Device, PropName,
                                                       PropSizeRet);
}

ol_result_t olIterateDevices(ol_device_iterate_cb_t Callback, void *UserData) {
  return mock::getMockLiboffload().olIterateDevices(Callback, UserData);
}

ol_result_t olDestroyProgram(ol_program_handle_t Program) {
  return mock::getMockLiboffload().olDestroyProgram(Program);
}

ol_result_t olCreateQueue(ol_device_handle_t Device, ol_queue_handle_t *Queue) {
  return mock::getMockLiboffload().olCreateQueue(Device, Queue);
}

ol_result_t olDestroyQueue(ol_queue_handle_t Queue) {
  return mock::getMockLiboffload().olDestroyQueue(Queue);
}

ol_result_t olSyncQueue(ol_queue_handle_t Queue) {
  return mock::getMockLiboffload().olSyncQueue(Queue);
}

ol_result_t olCreateProgram(ol_device_handle_t Device, const void *ProgData,
                            size_t ProgDataSize, ol_program_handle_t *Program) {
  return mock::getMockLiboffload().olCreateProgram(Device, ProgData,
                                                   ProgDataSize, Program);
}

ol_result_t olGetSymbol(ol_program_handle_t Program, const char *Name,
                        ol_symbol_kind_t Kind, ol_symbol_handle_t *Symbol) {
  return mock::getMockLiboffload().olGetSymbol(Program, Name, Kind, Symbol);
}

ol_result_t olIsValidBinary(ol_device_handle_t Device, const void *ProgData,
                            size_t ProgDataSize, bool *Valid) {
  return mock::getMockLiboffload().olIsValidBinary(Device, ProgData,
                                                   ProgDataSize, Valid);
}

ol_result_t olWaitEvents(ol_queue_handle_t Queue, ol_event_handle_t *Events,
                         size_t NumEvents) {
  return mock::getMockLiboffload().olWaitEvents(Queue, Events, NumEvents);
}

ol_result_t olLaunchKernel(ol_queue_handle_t Queue, ol_device_handle_t Device,
                           ol_symbol_handle_t Kernel,
                           const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                           const ol_kernel_launch_prop_t *Properties,
                           size_t NumArgs, void **ArgPtrs,
                           const size_t *ArgSizes) {
  return mock::getMockLiboffload().olLaunchKernel(Queue, Device, Kernel,
                                                  LaunchSizeArgs, Properties,
                                                  NumArgs, ArgPtrs, ArgSizes);
}

ol_result_t olSyncEvent(ol_event_handle_t Event) {
  return mock::getMockLiboffload().olSyncEvent(Event);
}

ol_result_t olMemcpy(ol_queue_handle_t Queue, void *DstPtr,
                     ol_device_handle_t DstDevice, const void *SrcPtr,
                     ol_device_handle_t SrcDevice, size_t Size) {
  return mock::getMockLiboffload().olMemcpy(Queue, DstPtr, DstDevice, SrcPtr,
                                            SrcDevice, Size);
}

ol_result_t olGetMemInfo(const void *Ptr, ol_mem_info_t PropName,
                         size_t PropSize, void *PropValue) {
  return mock::getMockLiboffload().olGetMemInfo(Ptr, PropName, PropSize,
                                                PropValue);
}

ol_result_t olCreateEvent(ol_queue_handle_t Queue, ol_event_flags_t Flags,
                          ol_event_handle_t *Event) {
  return mock::getMockLiboffload().olCreateEvent(Queue, Flags, Event);
}

ol_result_t olDestroyEvent(ol_event_handle_t Event) {
  return mock::getMockLiboffload().olDestroyEvent(Event);
}
