include(ExternalProject)
include(CheckCXXCompilerFlag)

#==============================================================================
# Build Google Benchmark for libc++
#==============================================================================

set(BENCHMARK_LIBCXX_COMPILE_FLAGS
    -Wno-unused-command-line-argument
    -nostdinc++
    -isystem ${LIBCXX_SOURCE_DIR}/include
    -L${LIBCXX_LIBRARY_DIR}
    -Wl,-rpath,${LIBCXX_LIBRARY_DIR}
    )
if (DEFINED LIBCXX_CXX_ABI_LIBRARY_PATH)
  list(APPEND BENCHMARK_LIBCXX_COMPILE_FLAGS
          -L${LIBCXX_CXX_ABI_LIBRARY_PATH}
          -Wl,-rpath,${LIBCXX_CXX_ABI_LIBRARY_PATH})
endif()
split_list(BENCHMARK_LIBCXX_COMPILE_FLAGS)

ExternalProject_Add(google-benchmark-libcxx
        EXCLUDE_FROM_ALL ON
        DEPENDS cxx
        PREFIX benchmark-libcxx
        SOURCE_DIR ${LIBCXX_SOURCE_DIR}/utils/google-benchmark
        INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/benchmark-libcxx
        CMAKE_CACHE_ARGS
          -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
          -DCMAKE_BUILD_TYPE:STRING=RELEASE
          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          -DCMAKE_CXX_FLAGS:STRING=${BENCHMARK_LIBCXX_COMPILE_FLAGS}
          -DBENCHMARK_USE_LIBCXX:BOOL=ON
          -DBENCHMARK_ENABLE_TESTING:BOOL=OFF)

#==============================================================================
# Build Google Benchmark for the native stdlib
#==============================================================================
if (LIBCXX_BUILD_BENCHMARK_NATIVE_STDLIB)
  ExternalProject_Add(google-benchmark-native
        EXCLUDE_FROM_ALL ON
        PREFIX benchmark-native
        SOURCE_DIR ${LIBCXX_SOURCE_DIR}/utils/google-benchmark
        INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/benchmark-native
        CMAKE_CACHE_ARGS
          -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
          -DCMAKE_BUILD_TYPE:STRING=RELEASE
          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          -DBENCHMARK_ENABLE_TESTING:BOOL=OFF)
endif()

#==============================================================================
# Benchmark tests configuration
#==============================================================================
add_custom_target(cxx-benchmarks)

set(BENCHMARK_LIBCXX_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/benchmark-libcxx)
set(BENCHMARK_NATIVE_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/benchmark-native)
set(BENCHMARK_TEST_COMPILE_FLAGS
    -std=c++14 -O2
    -I${BENCHMARK_LIBCXX_INSTALL}/include
)
set(BENCHMARK_TEST_LIBCXX_COMPILE_FLAGS
    -nostdinc++
    -isystem ${LIBCXX_SOURCE_DIR}/include
    ${BENCHMARK_TEST_COMPILE_FLAGS}
    -Wno-user-defined-literals
)
set(BENCHMARK_TEST_LIBCXX_LINK_FLAGS
    -nodefaultlibs
    -L${BENCHMARK_LIBCXX_INSTALL}/lib/
)
set(BENCHMARK_TEST_NATIVE_LINK_FLAGS
    -L${BENCHMARK_NATIVE_INSTALL}/lib/
)
split_list(BENCHMARK_TEST_COMPILE_FLAGS)
split_list(BENCHMARK_TEST_LIBCXX_COMPILE_FLAGS)
split_list(BENCHMARK_TEST_LIBCXX_LINK_FLAGS)
split_list(BENCHMARK_TEST_NATIVE_LINK_FLAGS)
macro(add_benchmark_test name source_file)
  set(libcxx_target ${name}_libcxx)
  add_executable(${libcxx_target} EXCLUDE_FROM_ALL ${source_file})
  add_dependencies(${libcxx_target} cxx google-benchmark-libcxx)
  add_dependencies(cxx-benchmarks ${libcxx_target})
  if (LIBCXX_ENABLE_SHARED)
    target_link_libraries(${libcxx_target} cxx_shared)
  else()
    target_link_libraries(${libcxx_target} cxx_static)
  endif()
  target_link_libraries(${libcxx_target} -lbenchmark)
  set_target_properties(${libcxx_target}
    PROPERTIES
          OUTPUT_NAME "${name}.libcxx.out"
          COMPILE_FLAGS "${BENCHMARK_TEST_LIBCXX_COMPILE_FLAGS}"
          LINK_FLAGS "${BENCHMARK_TEST_LIBCXX_LINK_FLAGS}")
  if (LIBCXX_BUILD_BENCHMARK_NATIVE_STDLIB)
    set(native_target ${name}_native)
    add_executable(${native_target} EXCLUDE_FROM_ALL ${source_file})
    add_dependencies(${native_target} google-benchmark-native)
    target_link_libraries(${native_target} -lbenchmark)
    if (LIBCXX_HAS_PTHREAD_LIB)
      target_link_libraries(${native_target} -pthread)
    endif()
    add_dependencies(cxx-benchmarks ${native_target})
    set_target_properties(${native_target}
      PROPERTIES
          OUTPUT_NAME "${name}.native.out"
          INCLUDE_DIRECTORIES ""
          COMPILE_FLAGS "${BENCHMARK_TEST_COMPILE_FLAGS}"
          LINK_FLAGS "${BENCHMARK_TEST_NATIVE_LINK_FLAGS}")
  endif()
endmacro()


#==============================================================================
# Register Benchmark tests
#==============================================================================
file(GLOB BENCHMARK_TESTS "*.bench.cpp")
foreach(test_path ${BENCHMARK_TESTS})
  get_filename_component(test_file "${test_path}" NAME)
  message(STATUS "Adding Benchmark: ${test_file}")
  string(REPLACE ".bench.cpp" "" test_name "${test_file}")
  add_benchmark_test(${test_name} ${test_file})
endforeach()
