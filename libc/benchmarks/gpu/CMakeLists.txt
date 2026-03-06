add_subdirectory(timing)

add_custom_target(gpu-benchmark)

function(add_benchmark benchmark_name)
  cmake_parse_arguments(
    "BENCHMARK"
    "" # Optional arguments
    "" # Single value arguments
    "LINK_LIBRARIES;DEPENDS" # Multi-value arguments
    ${ARGN}
  )

  if(NOT libc.src.time.clock IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS)
    message(FATAL_ERROR "target does not support clock")
  endif()
  add_libc_hermetic(
    ${benchmark_name}
    IS_GPU_BENCHMARK
    LINK_LIBRARIES
      LibcGpuBenchmark.hermetic
      ${BENCHMARK_LINK_LIBRARIES}
    DEPENDS
      libc.src.stdio.printf
      ${BENCHMARK_DEPENDS}
    ${BENCHMARK_UNPARSED_ARGUMENTS}
    COMPILE_OPTIONS
      -flto
  )
  get_fq_target_name(${benchmark_name} fq_target_name)
  set(fq_build_target_name ${fq_target_name}.__build__)

  add_dependencies(gpu-benchmark ${fq_target_name})
endfunction(add_benchmark)

function(add_benchmark_framework_library name)
  cmake_parse_arguments(
    "TEST_LIB"
    "" # No optional arguments
    "" # No single value arguments
    "SRCS;HDRS;DEPENDS" # Multi value arguments
    ${ARGN}
  )
  if(NOT TEST_LIB_SRCS)
    message(FATAL_ERROR "'add_benchmark_framework_library' requires SRCS; for "
                        "header only libraries, use 'add_header_library'")
  endif()

  foreach(lib IN ITEMS ${name}.hermetic)
    add_library(
      ${lib}
      STATIC
      EXCLUDE_FROM_ALL
      ${TEST_LIB_SRCS}
      ${TEST_LIB_HDRS}
    )
    target_include_directories(${lib} PRIVATE ${LIBC_SOURCE_DIR})
    if(TARGET libc.src.time.clock)
      target_compile_definitions(${lib} PRIVATE TARGET_SUPPORTS_CLOCK)
    endif()
  endforeach()

  _get_hermetic_test_compile_options(compile_options "")
  target_include_directories(${name}.hermetic PRIVATE ${LIBC_INCLUDE_DIR})
  target_compile_options(${name}.hermetic PRIVATE ${compile_options} -nostdinc++)

  if(TEST_LIB_DEPENDS)
    foreach(dep IN ITEMS ${TEST_LIB_DEPENDS})
      if(TARGET ${dep}.hermetic)
        add_dependencies(${name}.hermetic ${dep}.hermetic)
      else()
        add_dependencies(${name}.hermetic ${dep})
      endif()
    endforeach()
  endif()
endfunction()

add_benchmark_framework_library(
  LibcGpuBenchmark
  SRCS
    LibcGpuBenchmark.cpp
    LibcGpuBenchmarkMain.cpp
  HDRS
    LibcGpuBenchmark.h
    Random.h
  DEPENDS
    libc.benchmarks.gpu.timing.timing
    libc.hdr.stdint_proxy
    libc.src.__support.CPP.string
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.CPP.algorithm
    libc.src.__support.CPP.atomic
    libc.src.__support.CPP.array
    libc.src.__support.CPP.optional
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.nearest_integer_operations
    libc.src.__support.FPUtil.sqrt
    libc.src.__support.sign
    libc.src.__support.fixedvector
    libc.src.__support.GPU.utils
    libc.src.__support.time.gpu.time_utils
    libc.src.__support.macros.attributes
    libc.src.__support.macros.config
    libc.src.__support.macros.properties.types
    libc.src.stdio.printf
    libc.src.time.clock
)

add_subdirectory(src)
