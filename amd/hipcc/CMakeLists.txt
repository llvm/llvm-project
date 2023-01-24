cmake_minimum_required(VERSION 3.13.4)

project(hipcc VERSION "1.0.0" LANGUAGES C CXX)
set(hipcc_NAME "${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(ADDITIONAL_SHARED_LIBRARIES_TO_LINK
  libstdc++fs.so)

set(HIP_VERSION_MAJOR 4 PARENT_SCOPE)
set(HIP_VERSION_MINOR 4 PARENT_SCOPE)
set(HIP_VERSION_PATCH 4 PARENT_SCOPE)

set(HIPCC_BIN
  hipcc.bin)
set(HIPCC_SOURCES
  src/hipBin.cpp)

set(HIPCONFIG_BIN
  hipconfig.bin)
set(HIPCONFIG_SOURCES
  src/hipBin.cpp)

add_executable(${HIPCC_BIN} ${HIPCC_SOURCES})
if(NOT WIN32)
  # C++17 does not require std lib linking.
  target_link_libraries(${HIPCC_BIN} ${ADDITIONAL_SHARED_LIBRARIES_TO_LINK})
endif()

add_executable(${HIPCONFIG_BIN} ${HIPCONFIG_SOURCES})
if(NOT WIN32)
  # C++17 does not require std lib linking.
  target_link_libraries(${HIPCONFIG_BIN} ${ADDITIONAL_SHARED_LIBRARIES_TO_LINK})
endif()
