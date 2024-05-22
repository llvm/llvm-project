add_library(gpu_loader OBJECT Main.cpp)
target_include_directories(gpu_loader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LIBC_SOURCE_DIR}/include
  ${LIBC_SOURCE_DIR}
)

find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(hsa-runtime64_FOUND)
  add_subdirectory(amdgpu)
else()
  message(STATUS "Skipping HSA loader for gpu target, no HSA was detected")
endif()

find_package(CUDAToolkit QUIET)
# The CUDA loader requires LLVM to traverse the ELF image for symbols.
find_package(LLVM QUIET)
if(CUDAToolkit_FOUND AND LLVM_FOUND AND
   "${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "11.2")
  add_subdirectory(nvptx)
else()
  if("${CUDAToolkit_VERSION}" VERSION_LESS "11.2")
    message(WARNING 
      "Skipping CUDA loader for gpu target, CUDA must be version 11.2 or later.
       Found CUDA Version ${CUDAToolkit_VERSION}")
  else()
    message(STATUS "Skipping CUDA loader for gpu target, no CUDA was detected")
  endif()
endif()

# Add a custom target to be used for testing.
set(LIBC_GPU_LOADER_EXECUTABLE "" CACHE STRING "Overriding binary for the GPU loader.")
if(LIBC_GPU_LOADER_EXECUTABLE)
  add_custom_target(libc.utils.gpu.loader)
  set_target_properties(
    libc.utils.gpu.loader
    PROPERTIES
      EXECUTABLE "${LIBC_GPU_LOADER_EXECUTABLE}"
  )
elseif(TARGET amdhsa_loader AND LIBC_GPU_TARGET_ARCHITECTURE_IS_AMDGPU)
  add_custom_target(libc.utils.gpu.loader)
  add_dependencies(libc.utils.gpu.loader amdhsa_loader)
  set_target_properties(
    libc.utils.gpu.loader
    PROPERTIES
      EXECUTABLE "$<TARGET_FILE:amdhsa_loader>"
  )
elseif(TARGET nvptx_loader AND LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
  add_custom_target(libc.utils.gpu.loader)
  add_dependencies(libc.utils.gpu.loader nvptx_loader)
  set_target_properties(
    libc.utils.gpu.loader
    PROPERTIES
      EXECUTABLE "$<TARGET_FILE:nvptx_loader>"
  )
endif()
