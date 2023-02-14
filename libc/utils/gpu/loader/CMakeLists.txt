add_library(gpu_loader OBJECT Main.cpp)
target_include_directories(gpu_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(hsa-runtime64_FOUND)
  add_subdirectory(amdgpu)
endif()
