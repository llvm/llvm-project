#==============================================================================
# Build Google Benchmark
#==============================================================================

include(ExternalProject)
set(BENCHMARK_COMPILE_FLAGS
    -Wno-unused-command-line-argument
    -nostdinc++
    -isystem "${LIBCXX_GENERATED_INCLUDE_DIR}"
    -L${LIBCXX_LIBRARY_DIR}
    -Wl,-rpath,${LIBCXX_LIBRARY_DIR}
    ${SANITIZER_FLAGS}
    )
if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  list(APPEND BENCHMARK_COMPILE_FLAGS
    -isystem "${LIBCXX_GENERATED_INCLUDE_TARGET_DIR}")
endif()
if (DEFINED LIBCXX_CXX_ABI_LIBRARY_PATH)
  list(APPEND BENCHMARK_COMPILE_FLAGS
          -L${LIBCXX_CXX_ABI_LIBRARY_PATH}
          -Wl,-rpath,${LIBCXX_CXX_ABI_LIBRARY_PATH})
endif()
split_list(BENCHMARK_COMPILE_FLAGS)

set(BENCHMARK_CXX_LIBRARIES)
list(APPEND BENCHMARK_CXX_LIBRARIES c++)
if (NOT LIBCXX_ENABLE_SHARED)
  list(APPEND BENCHMARK_CXX_LIBRARIES c++abi)
endif()

ExternalProject_Add(google-benchmark
        EXCLUDE_FROM_ALL ON
        DEPENDS cxx cxx-headers
        PREFIX google-benchmark
        SOURCE_DIR ${LLVM_THIRD_PARTY_DIR}/benchmark
        INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/google-benchmark
        CMAKE_CACHE_ARGS
          -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
          -DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}
          -DCMAKE_BUILD_TYPE:STRING=RELEASE
          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          -DCMAKE_CXX_FLAGS:STRING=${BENCHMARK_COMPILE_FLAGS}
          -DBENCHMARK_USE_LIBCXX:BOOL=ON
          -DBENCHMARK_ENABLE_TESTING:BOOL=OFF
          -DBENCHMARK_CXX_LIBRARIES:STRING=${BENCHMARK_CXX_LIBRARIES})

add_dependencies(cxx-test-depends google-benchmark)
