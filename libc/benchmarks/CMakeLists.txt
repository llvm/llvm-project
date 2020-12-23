find_package(Threads)

include(ExternalProject)

set(LLVM_LINK_COMPONENTS Support)

#==============================================================================
# Build Google Benchmark
#==============================================================================
set(GOOGLE_BENCHMARK_TARGET_FLAGS ${BENCHMARK_DIALECT_FLAG})
if (LIBCXX_BENCHMARK_GCC_TOOLCHAIN)
  set(GOOGLE_BENCHMARK_TARGET_FLAGS
      -gcc-toolchain ${LIBCXX_BENCHMARK_GCC_TOOLCHAIN})
endif()
string(REPLACE ";" " " GOOGLE_BENCHMARK_TARGET_FLAGS "${GOOGLE_BENCHMARK_TARGET_FLAGS}")

ExternalProject_Add(google-benchmark
    EXCLUDE_FROM_ALL ON
    PREFIX google-benchmark
    SOURCE_DIR ${LIBC_SOURCE_DIR}/../llvm/utils/benchmark
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/google-benchmark
    CMAKE_CACHE_ARGS
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
        -DCMAKE_CXX_FLAGS:STRING=${GOOGLE_BENCHMARK_TARGET_FLAGS}
        -DCMAKE_CXX_STANDARD:STRING=14
        -DCMAKE_BUILD_TYPE:STRING=RELEASE
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DBENCHMARK_ENABLE_TESTING:BOOL=OFF)

set(GOOGLE_BENCHMARK_LIBC_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/google-benchmark)
set(GOOGLE_BENCHMARK_LINK_FLAGS    -L${GOOGLE_BENCHMARK_LIBC_INSTALL}/lib/)

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
    gtest_main
    gtest
    ${LIBC_BENCHMARKS_UNITTEST_DEPENDS}
  )

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

function(fix_rtti target)
    # TODO: Make this portable and inline with rtti mode from llvm/
    target_compile_options(${target} PUBLIC -fno-rtti)
endfunction()

# libc-benchmark
add_library(libc-benchmark
    STATIC
    EXCLUDE_FROM_ALL
    LibcBenchmark.cpp
    LibcBenchmark.h
)
add_dependencies(libc-benchmark google-benchmark)
target_include_directories(libc-benchmark
    SYSTEM PUBLIC
    "${GOOGLE_BENCHMARK_LIBC_INSTALL}/include"
)
target_link_libraries(libc-benchmark
    PUBLIC
    "${GOOGLE_BENCHMARK_LINK_FLAGS}" # FIXME: Move to `target_link_options`
    -lbenchmark                      # FIXME: Move to `target_link_options`
    LLVMSupport
    Threads::Threads
)
fix_rtti(libc-benchmark)

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
    MemorySizeDistributions.cpp
    MemorySizeDistributions.h
)
target_link_libraries(libc-memory-benchmark
    PUBLIC
    libc-benchmark
)
fix_rtti(libc-memory-benchmark)

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
fix_rtti(json)

add_libc_benchmark_unittest(json-test
    SRCS JSONTest.cpp
    DEPENDS json
)

#==============================================================================
# Benchmarking tool
#==============================================================================

add_executable(libc-benchmark-main
    EXCLUDE_FROM_ALL
    LibcMemoryBenchmarkMain.cpp
)
foreach(entrypoint_target libc.src.string.memcpy libc.src.string.memset)
    get_target_property(entrypoint_object_file ${entrypoint_target} "OBJECT_FILE_RAW")
    target_link_libraries(libc-benchmark-main PUBLIC json ${entrypoint_object_file})
endforeach()
