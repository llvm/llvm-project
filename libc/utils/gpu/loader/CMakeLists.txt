add_library(gpu_loader OBJECT Main.cpp)

target_include_directories(gpu_loader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LIBC_SOURCE_DIR}/include
  ${LIBC_SOURCE_DIR}
)

# This utility needs to be compiled for the host system when cross compiling.
if(LLVM_RUNTIMES_TARGET OR LIBC_TARGET_TRIPLE)
  target_compile_options(gpu_loader PUBLIC --target=${LLVM_HOST_TRIPLE})
  target_link_libraries(gpu_loader PUBLIC "--target=${LLVM_HOST_TRIPLE}")
endif()

find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(hsa-runtime64_FOUND AND LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  add_subdirectory(amdgpu)
elseif(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  message(STATUS "Skipping HSA loader for gpu target, no HSA was detected")
endif()

# The CUDA loader requires LLVM to traverse the ELF image for symbols.
find_package(LLVM QUIET)
if(CUDAToolkit_FOUND AND LLVM_FOUND AND LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  add_subdirectory(nvptx)
elseif(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  message(STATUS "Skipping CUDA loader for gpu target, no CUDA was detected")
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
elseif(TARGET amdhsa-loader AND LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  add_custom_target(libc.utils.gpu.loader)
  add_dependencies(libc.utils.gpu.loader amdhsa-loader)
  set_target_properties(
    libc.utils.gpu.loader
    PROPERTIES
      TARGET amdhsa-loader
      EXECUTABLE "$<TARGET_FILE:amdhsa-loader>"
  )
elseif(TARGET nvptx-loader AND LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  add_custom_target(libc.utils.gpu.loader)
  add_dependencies(libc.utils.gpu.loader nvptx-loader)
  set_target_properties(
    libc.utils.gpu.loader
    PROPERTIES
      TARGET nvptx-loader
      EXECUTABLE "$<TARGET_FILE:nvptx-loader>"
  )
endif()

if(TARGET libc.utils.gpu.loader)
  get_target_property(gpu_loader_tgt libc.utils.gpu.loader "TARGET")
  if(gpu_loader_tgt)
    install(TARGETS ${gpu_loader_tgt}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT libc)
  endif()
endif()
