add_custom_target(libc-gpu-math-benchmarks)

set(math_benchmark_flags "")
if(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  if(CUDAToolkit_FOUND)
    set(libdevice_path ${CUDAToolkit_BIN_DIR}/../nvvm/libdevice/libdevice.10.bc)
    if (EXISTS ${libdevice_path})
      list(APPEND math_benchmark_flags
        "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${libdevice_path}")
      # Compile definition needed so the benchmark knows to register
      # NVPTX benchmarks.
      list(APPEND math_benchmark_flags "-DNVPTX_MATH_FOUND=1")
    endif()
  endif()
endif()

if(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  find_package(AMDDeviceLibs QUIET HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
  if(AMDDeviceLibs_FOUND)
    get_target_property(ocml_path ocml IMPORTED_LOCATION)
    list(APPEND math_benchmark_flags
        "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${ocml_path}")
    list(APPEND math_benchmark_flags "-DAMDGPU_MATH_FOUND=1")
  endif()
endif()

add_benchmark(
  sin_benchmark
  SUITE
    libc-gpu-math-benchmarks
  SRCS
    sin_benchmark.cpp
  DEPENDS
    libc.src.math.sin
    libc.src.math.sinf
    libc.src.stdlib.srand
    libc.src.stdlib.rand
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.array
  COMPILE_OPTIONS
    ${math_benchmark_flags}
  LOADER_ARGS
    --threads 64
)

add_benchmark(
  atan2_benchmark
  SUITE
    libc-gpu-math-benchmarks
  SRCS
    atan2_benchmark.cpp
  DEPENDS
    libc.src.math.atan2
    libc.src.stdlib.srand
    libc.src.stdlib.rand
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.array
  COMPILE_OPTIONS
    ${math_benchmark_flags}
  LOADER_ARGS
    --threads 64
)
