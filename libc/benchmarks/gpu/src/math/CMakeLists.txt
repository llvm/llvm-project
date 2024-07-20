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
  COMPILE_OPTIONS
    ${nvptx_math_found}
    ${nvptx_bitcode_link_flags}
  LOADER_ARGS
    --threads 64
)