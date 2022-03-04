find_package(Threads)

set(LLVM_LINK_COMPONENTS Support)

#==============================================================================
# Add Unit Testing Support
#==============================================================================

function(add_libc_benchmark_unittest target_name)
  if(NOT LLVM_INCLUDE_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    "LIBC_BENCHMARKS_UNITTEST"
    "" # No optional arguments
    "SUITE" # Single value arguments
    "SRCS;DEPENDS" # Multi-value arguments
    ${ARGN}
  )

  add_executable(${target_name}
    EXCLUDE_FROM_ALL
    ${LIBC_BENCHMARKS_UNITTEST_SRCS}
  )
  target_link_libraries(${target_name}
    PRIVATE
    llvm_gtest_main
    llvm_gtest
    ${LIBC_BENCHMARKS_UNITTEST_DEPENDS}
  )
  llvm_update_compile_flags(${target_name})

  add_custom_command(
    TARGET ${target_name}
    POST_BUILD
    COMMAND $<TARGET_FILE:${target_name}>
  )
  add_dependencies(libc-benchmark-util-tests ${target_name})
endfunction()

#==============================================================================
# Build Google Benchmark for libc
#==============================================================================

add_custom_target(libc-benchmark-util-tests)

# libc-benchmark
add_library(libc-benchmark
    STATIC
    EXCLUDE_FROM_ALL
    LibcBenchmark.cpp
    LibcBenchmark.h
)
add_dependencies(libc-benchmark benchmark)
target_link_libraries(libc-benchmark
    PUBLIC
    benchmark
    LLVMSupport
    Threads::Threads
)
llvm_update_compile_flags(libc-benchmark)

add_libc_benchmark_unittest(libc-benchmark-test
    SRCS LibcBenchmarkTest.cpp
    DEPENDS libc-benchmark
)

# libc-memory-benchmark
add_library(libc-memory-benchmark
    STATIC
    EXCLUDE_FROM_ALL
    LibcMemoryBenchmark.cpp
    LibcMemoryBenchmark.h
    LibcFunctionPrototypes.h
    MemorySizeDistributions.cpp
    MemorySizeDistributions.h
)
target_include_directories(libc-memory-benchmark
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(libc-memory-benchmark
    PUBLIC
    libc-benchmark
)
llvm_update_compile_flags(libc-memory-benchmark)

add_libc_benchmark_unittest(libc-memory-benchmark-test
    SRCS LibcMemoryBenchmarkTest.cpp
    DEPENDS libc-memory-benchmark
)

# json
add_library(json
    STATIC
    EXCLUDE_FROM_ALL
    JSON.cpp
    JSON.h
)
target_link_libraries(json PUBLIC libc-memory-benchmark)
llvm_update_compile_flags(json)

add_libc_benchmark_unittest(json-test
    SRCS JSONTest.cpp
    DEPENDS json
)

#==============================================================================
# Benchmarking tool
#==============================================================================

# Benchmark all implementations that can run on the target CPU.
function(add_libc_multi_impl_benchmark name)
  get_property(fq_implementations GLOBAL PROPERTY ${name}_implementations)
  foreach(fq_config_name IN LISTS fq_implementations)
    get_target_property(required_cpu_features ${fq_config_name} REQUIRE_CPU_FEATURES)
    cpu_supports(can_run "${required_cpu_features}")
    if(can_run)
        set(benchmark_name ${fq_config_name}_benchmark)
        add_executable(${benchmark_name}
            EXCLUDE_FROM_ALL
            LibcMemoryBenchmarkMain.cpp
        )
        get_target_property(entrypoint_object_file ${fq_config_name} "OBJECT_FILE_RAW")
        target_link_libraries(${benchmark_name} PUBLIC json ${entrypoint_object_file})
        string(TOUPPER ${name} name_upper)
        target_compile_definitions(${benchmark_name} PRIVATE "-DLIBC_BENCHMARK_FUNCTION_${name_upper}=__llvm_libc::${name}" "-DLIBC_BENCHMARK_FUNCTION_NAME=\"${fq_config_name}\"")
        llvm_update_compile_flags(${benchmark_name})
    else()
      message(STATUS "Skipping benchmark for '${fq_config_name}' insufficient host cpu features '${required_cpu_features}'")
    endif()
  endforeach()
endfunction()

add_libc_multi_impl_benchmark(bcmp)
add_libc_multi_impl_benchmark(bzero)
add_libc_multi_impl_benchmark(memcmp)
add_libc_multi_impl_benchmark(memcpy)
add_libc_multi_impl_benchmark(memmove)
add_libc_multi_impl_benchmark(memset)

#==============================================================================
# Google Benchmarking tool
#==============================================================================

# This target uses the Google Benchmark facility to report throughput for llvm
# libc memory functions compiled for the host machine. This is useful to
# continuously monitor the performance of the memory functions.
add_executable(libc.benchmarks.memory_functions.opt_host
  EXCLUDE_FROM_ALL
  LibcMemoryGoogleBenchmarkMain.cpp
  LibcDefaultImplementations.cpp
)
target_link_libraries(libc.benchmarks.memory_functions.opt_host
  PRIVATE
  libc-memory-benchmark
  libc.src.string.memcmp_opt_host
  libc.src.string.bcmp_opt_host
  libc.src.string.memcpy_opt_host
  libc.src.string.memset_opt_host
  libc.src.string.bzero_opt_host
  libc.src.string.memmove_opt_host
  benchmark_main
)
llvm_update_compile_flags(libc.benchmarks.memory_functions.opt_host)

add_subdirectory(automemcpy)
