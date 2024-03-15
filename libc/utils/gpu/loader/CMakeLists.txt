add_library(gpu_loader OBJECT Main.cpp)

target_include_directories(gpu_loader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LIBC_SOURCE_DIR}/include
  ${LIBC_SOURCE_DIR}
)

find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(hsa-runtime64_FOUND)
  add_subdirectory(amdgpu)
endif()

# The CUDA loader requires LLVM to traverse the ELF image for symbols.
find_package(CUDAToolkit 11.2 QUIET)
if(CUDAToolkit_FOUND)
  add_subdirectory(nvptx)
endif()

if(TARGET amdhsa-loader AND LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
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

foreach(gpu_loader_tgt amdhsa-loader nvptx-loader)
  if(TARGET ${gpu_loader_tgt})
    install(TARGETS ${gpu_loader_tgt}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT libc)
  endif()
endforeach()
