add_subdirectory(timing)

add_custom_target(gpu-benchmark)

function(add_benchmark benchmark_name)
  cmake_parse_arguments(
    "BENCHMARK"
    "" # Optional arguments
    "" # Single value arguments
    "LINK_LIBRARIES" # Multi-value arguments
    ${ARGN}
  )
  if(NOT libc.src.time.clock IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS)
    message(FATAL_ERROR "target does not support clock")
  endif()
  add_libc_hermetic(
    ${benchmark_name}
    IS_BENCHMARK
    LINK_LIBRARIES
      LibcGpuBenchmark.hermetic
      ${BENCHMARK_LINK_LIBRARIES}
    ${BENCHMARK_UNPARSED_ARGUMENTS}
  )
  get_fq_target_name(${benchmark_name} fq_target_name)
  set(fq_build_target_name ${fq_target_name}.__build__)

  set(grep_res_usage_cmd grep -A 3 "${benchmark_name}")
  set(res_usage_cmd cuobjdump -res-usage $<TARGET_FILE:${fq_build_target_name}> | ${grep_res_usage_cmd})
  add_custom_command(
    OUTPUT ${fq_target_name}-cmd APPEND
    COMMAND ${res_usage_cmd}
    COMMAND_EXPAND_LISTS
    COMMENT "Reading resource usage for ${benchmark_name}"
    ${LIBC_HERMETIC_TEST_JOB_POOL}
  )
  # add_dependencies(${fq_target_name} ${fq_target_name}_resources)
  # set_source_files_properties(${fq_target_name}_resources
  #   PROPERTIES
  #     SYMBOLIC "TRUE"
  # )

  add_dependencies(gpu-benchmark ${fq_target_name})
endfunction(add_benchmark)

add_unittest_framework_library(
  LibcGpuBenchmark
  SRCS
    LibcGpuBenchmark.cpp
    LibcGpuBenchmarkMain.cpp
    BenchmarkLogger.cpp
  HDRS
    LibcGpuBenchmark.h
    BenchmarkLogger.h
  DEPENDS
    libc.src.__support.big_int
    libc.src.__support.c_string
    libc.src.__support.CPP.string
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.CPP.functional
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.algorithm
    libc.src.__support.fixed_point.fx_rep
    libc.src.__support.macros.properties.types
    libc.src.__support.OSUtil.osutil
    libc.src.__support.uint128
    libc.src.__support.FPUtil.sqrt
    libc.src.__support.fixedvector
    libc.src.time.clock
    libc.benchmarks.gpu.timing.timing
)

add_subdirectory(src)
