cmake_minimum_required(VERSION 3.4.3)

project(XcodeConfig C CXX)

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_SOURCE_DIR}/../modules"
  )

set(LLDB_CONFIG_HEADER_INPUT
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/lldb/Host/Config.h.cmake)

include(LLDBGenerateConfig)
