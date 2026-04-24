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
