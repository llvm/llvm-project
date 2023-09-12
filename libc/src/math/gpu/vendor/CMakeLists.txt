find_package(AMDDeviceLibs QUIET HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(AMDDeviceLibs_FOUND)
  add_subdirectory(amdgpu)
  set(LIBC_GPU_VENDOR "amdgpu" CACHE INTERNAL "")
  message(STATUS "Found the ROCm device library. Implementations falling back "
                 "to the vendor libraries will be resolved statically.")
  get_target_property(ocml_path ocml IMPORTED_LOCATION)
  list(APPEND bitcode_link_flags
       "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${ocml_path}")
else()
  message(STATUS "Could not find the ROCm device library. Unimplemented "
                 "functions will be an external reference to the vendor libraries.")
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  add_subdirectory(nvptx)
  set(LIBC_GPU_VENDOR "nvptx" CACHE INTERNAL "")
  set(libdevice_path ${CUDAToolkit_BIN_DIR}/../nvvm/libdevice/libdevice.10.bc)
  if (EXISTS ${libdevice_path})
    message(STATUS "Found the CUDA device library. Implementations falling back "
                   "to the vendor libraries will be resolved statically.")
    list(APPEND bitcode_link_flags
         "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${libdevice_path}")
  endif()
else()
  message(STATUS "Could not find the ROCm device library. Unimplemented "
                 "functions will be an external reference to the vendor libraries.")
endif()

# FIXME: We need a way to pass the library to only the NVTPX / AMDGPU build.
# This shouldn't cause issues because we only link in needed symbols, but it
# will link in identity metadata from both libraries. This silences the warning.
list(APPEND bitcode_link_flags "-Wno-linker-warnings")