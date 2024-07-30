add_custom_target(libc-gpu-math-benchmarks)

if(CUDAToolkit_FOUND)
  set(libdevice_path ${CUDAToolkit_BIN_DIR}/../nvvm/libdevice/libdevice.10.bc)
  if (EXISTS ${libdevice_path})
    set(nvptx_bitcode_link_flags
        "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${libdevice_path}")
    # Compile definition needed so the benchmark knows to register
    # NVPTX benchmarks.
    set(nvptx_math_found "-DNVPTX_MATH_FOUND=1")
  endif()
endif()

find_package(AMDDeviceLibs QUIET HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(AMDDeviceLibs_FOUND)
  get_target_property(ocml_path ocml IMPORTED_LOCATION)
  set(amdgpu_bitcode_link_flags
      "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${ocml_path}")
  set(amdgpu_math_found "-DAMDGPU_MATH_FOUND=1")
endif()

add_benchmark(
  sin_benchmark
  SUITE
    libc-gpu-math-benchmarks
  SRCS
    sin_benchmark.cpp
  DEPENDS
    libc.src.math.sin
    libc.src.stdlib.srand
    libc.src.stdlib.rand
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.array
  COMPILE_OPTIONS
    ${nvptx_math_found}
    ${nvptx_bitcode_link_flags}
    ${amdgpu_math_found}
    ${amdgpu_bitcode_link_flags}
  LOADER_ARGS
    --threads 64
)
