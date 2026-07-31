# The CPU build depends on Google benchmark.
if(NOT LIBC_INCLUDE_BENCHMARKS)
  return()
endif()

if(LIBC_TARGET_OS_IS_GPU)
  add_subdirectory(gpu)
  return()
endif()

find_package(Threads)

option(LIBC_BENCHMARKS_HAS_LLVM_SUPPORT "Avoid LLVM Support dependency and reporting CPU details" ON)

if(NOT TARGET LLVMSupport OR NOT TARGET LLVMTargetParser)
  set(LIBC_BENCHMARKS_HAS_LLVM_SUPPORT OFF CACHE BOOL "Avoid LLVM Support dependency and reporting CPU details" FORCE)
endif()

if(LIBC_BENCHMARKS_HAS_LLVM_SUPPORT)
  set(LLVM_LINK_COMPONENTS
    Support
    TargetParser
    )
endif()

#==============================================================================
# Add Unit Testing Support
#==============================================================================

if(COMMAND build_gtest)
  build_gtest()
elseif(COMMAND make_gtest_target)
  make_gtest_target()
endif()

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
    default_gtest_main
    default_gtest
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

include(ExternalProject)
ExternalProject_Add(google-benchmark-libc
        EXCLUDE_FROM_ALL ON
        PREFIX google-benchmark-libc
        SOURCE_DIR ${LLVM_THIRD_PARTY_DIR}/benchmark
        INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/google-benchmark-libc
        BUILD_BYPRODUCTS
          ${CMAKE_CURRENT_BINARY_DIR}/google-benchmark-libc/lib/libbenchmark.a
          ${CMAKE_CURRENT_BINARY_DIR}/google-benchmark-libc/lib/libbenchmark_main.a
        CMAKE_CACHE_ARGS
          -DBENCHMARK_ENABLE_EXCEPTIONS:BOOL=OFF
          -DBENCHMARK_ENABLE_LTO:BOOL=OFF
          -DBENCHMARK_ENABLE_TESTING:BOOL=OFF
          -DBENCHMARK_ENABLE_WERROR:BOOL=${LLVM_ENABLE_WERROR}
          -DBENCHMARK_FORCE_WERROR:BOOL=OFF
          -DBENCHMARK_USE_LIBCXX:BOOL=OFF
          -DCMAKE_BUILD_TYPE:STRING=Release

          -DCMAKE_SYSTEM_NAME:STRING=${CMAKE_SYSTEM_NAME}
          -DCMAKE_SYSTEM_PROCESSOR:STRING=${CMAKE_SYSTEM_PROCESSOR}
          -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
          -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
          -DCMAKE_FIND_ROOT_PATH:STRING=${CMAKE_FIND_ROOT_PATH}

          -DBUILD_SHARED_LIBS:BOOL=OFF
          -DCMAKE_EXE_LINKER_FLAGS:STRING=-static

          -DCMAKE_CXX_STANDARD:STRING=14
          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        )

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/google-benchmark-libc/include")

if(NOT TARGET benchmark::benchmark)
  add_library(benchmark::benchmark STATIC IMPORTED)
  set_target_properties(benchmark::benchmark PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/google-benchmark-libc/lib/libbenchmark.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/google-benchmark-libc/include"
  )
  add_dependencies(benchmark::benchmark google-benchmark-libc)
endif()

if(NOT TARGET benchmark_main)
  add_library(benchmark_main STATIC IMPORTED)
  set_target_properties(benchmark_main PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/google-benchmark-libc/lib/libbenchmark_main.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/google-benchmark-libc/include"
  )
  add_dependencies(benchmark_main google-benchmark-libc)
endif()

add_custom_target(libc-benchmark-util-tests)

# libc-benchmark
add_library(libc-benchmark
    STATIC
    EXCLUDE_FROM_ALL
    LibcBenchmark.cpp
    LibcBenchmark.h
)

target_include_directories(libc-benchmark
    PUBLIC ${LLVM_INCLUDE_DIR} ${LLVM_MAIN_INCLUDE_DIR}
)
target_link_libraries(libc-benchmark
    PUBLIC
    benchmark::benchmark
    Threads::Threads
)
if(LIBC_BENCHMARKS_HAS_LLVM_SUPPORT)
  target_link_libraries(libc-benchmark
      PUBLIC
      LLVMSupport
      LLVMTargetParser
  )
  target_compile_definitions(libc-benchmark PUBLIC "LIBC_BENCHMARKS_HAS_LLVM_SUPPORT")
else()
  target_compile_definitions(libc-benchmark PUBLIC "LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1")
endif()
add_dependencies(libc-benchmark google-benchmark-libc)
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
    ${LIBC_SOURCE_DIR}
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

if(LIBC_BENCHMARKS_HAS_LLVM_SUPPORT)
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
          target_compile_definitions(${benchmark_name} PRIVATE "-DLIBC_BENCHMARK_FUNCTION_${name_upper}=LIBC_NAMESPACE::${name}" "-DLIBC_BENCHMARK_FUNCTION_NAME=\"${fq_config_name}\"")
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
endif()

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
  libc.src.string.memcmp_opt_host.__internal__
  libc.src.string.memcpy_opt_host.__internal__
  libc.src.string.memmove_opt_host.__internal__
  libc.src.string.memset_opt_host.__internal__
  libc.src.strings.bcmp_opt_host.__internal__
  libc.src.strings.bzero_opt_host.__internal__
  benchmark_main
)
llvm_update_compile_flags(libc.benchmarks.memory_functions.opt_host)
add_dependencies(libc.benchmarks.memory_functions.opt_host google-benchmark-libc)

if(LIBC_TYPES_HAS_FLOAT16)
  add_executable(libc.benchmarks.rsqrtf16.opt_host
    EXCLUDE_FROM_ALL
    LibcRsqrtf16GoogleBenchmarkMain.cpp
  )
  target_include_directories(libc.benchmarks.rsqrtf16.opt_host
    PRIVATE
    ${LIBC_SOURCE_DIR}
    ${LIBC_INCLUDE_DIR}
  )
  target_link_libraries(libc.benchmarks.rsqrtf16.opt_host
    PRIVATE
    benchmark_main
    libc-benchmark
    libc.src.errno.errno.__internal__
    libc.src.__support.math.rsqrtf16
  )
  llvm_update_compile_flags(libc.benchmarks.rsqrtf16.opt_host)
 endif()
