# Enable the tests

find_package(Threads REQUIRED)

macro(compile_benchmark_test name)
  add_executable(${name} "${name}.cc")
  target_link_libraries(${name} benchmark ${CMAKE_THREAD_LIBS_INIT})
endmacro(compile_benchmark_test)

# Demonstration executable
compile_benchmark_test(benchmark_test)
add_test(benchmark benchmark_test --benchmark_min_time=0.01)

compile_benchmark_test(filter_test)
macro(add_filter_test name filter expect)
  add_test(${name} filter_test --benchmark_min_time=0.01 --benchmark_filter=${filter} ${expect})
  add_test(${name}_list_only filter_test --benchmark_list_tests --benchmark_filter=${filter} ${expect})
endmacro(add_filter_test)

add_filter_test(filter_simple "Foo" 3)
add_filter_test(filter_suffix "BM_.*" 4)
add_filter_test(filter_regex_all ".*" 5)
add_filter_test(filter_regex_blank "" 5)
add_filter_test(filter_regex_none "monkey" 0)
add_filter_test(filter_regex_wildcard ".*Foo.*" 3)
add_filter_test(filter_regex_begin "^BM_.*" 4)
add_filter_test(filter_regex_begin2 "^N" 1)
add_filter_test(filter_regex_end ".*Ba$" 1)

compile_benchmark_test(options_test)
add_test(options_benchmarks options_test --benchmark_min_time=0.01)

compile_benchmark_test(basic_test)
add_test(basic_benchmark basic_test --benchmark_min_time=0.01)

compile_benchmark_test(diagnostics_test)
add_test(diagnostics_test diagnostics_test --benchmark_min_time=0.01)

compile_benchmark_test(skip_with_error_test)
add_test(skip_with_error_test skip_with_error_test --benchmark_min_time=0.01)

compile_benchmark_test(donotoptimize_test)
add_test(donotoptimize_test donotoptimize_test --benchmark_min_time=0.01)

compile_benchmark_test(fixture_test)
add_test(fixture_test fixture_test --benchmark_min_time=0.01)

compile_benchmark_test(register_benchmark_test)
add_test(register_benchmark_test register_benchmark_test --benchmark_min_time=0.01)

compile_benchmark_test(map_test)
add_test(map_test map_test --benchmark_min_time=0.01)

compile_benchmark_test(multiple_ranges_test)
add_test(multiple_ranges_test multiple_ranges_test --benchmark_min_time=0.01)

compile_benchmark_test(reporter_output_test)
add_test(reporter_output_test reporter_output_test --benchmark_min_time=0.01)

check_cxx_compiler_flag(-std=c++03 BENCHMARK_HAS_CXX03_FLAG)
if (BENCHMARK_HAS_CXX03_FLAG)
  set(CXX03_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-std=c++11" "-std=c++03" CXX03_FLAGS "${CXX03_FLAGS}")
  string(REPLACE "-std=c++0x" "-std=c++03" CXX03_FLAGS "${CXX03_FLAGS}")

  compile_benchmark_test(cxx03_test)
  set_target_properties(cxx03_test
      PROPERTIES COMPILE_FLAGS "${CXX03_FLAGS}")
  add_test(cxx03 cxx03_test --benchmark_min_time=0.01)
endif()

compile_benchmark_test(complexity_test)
add_test(complexity_benchmark complexity_test --benchmark_min_time=0.01)

# Add the coverage command(s)
if(CMAKE_BUILD_TYPE)
  string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOWER)
endif()
if (${CMAKE_BUILD_TYPE_LOWER} MATCHES "coverage")
  find_program(GCOV gcov)
  find_program(LCOV lcov)
  find_program(GENHTML genhtml)
  find_program(CTEST ctest)
  if (GCOV AND LCOV AND GENHTML AND CTEST AND HAVE_CXX_FLAG_COVERAGE)
    add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/lcov/index.html
      COMMAND ${LCOV} -q -z -d .
      COMMAND ${LCOV} -q --no-external -c -b "${CMAKE_SOURCE_DIR}" -d . -o before.lcov -i
      COMMAND ${CTEST} --force-new-ctest-process
      COMMAND ${LCOV} -q --no-external -c -b "${CMAKE_SOURCE_DIR}" -d . -o after.lcov
      COMMAND ${LCOV} -q -a before.lcov -a after.lcov --output-file final.lcov
      COMMAND ${LCOV} -q -r final.lcov "'${CMAKE_SOURCE_DIR}/test/*'" -o final.lcov
      COMMAND ${GENHTML} final.lcov -o lcov --demangle-cpp --sort -p "${CMAKE_BINARY_DIR}" -t benchmark
      DEPENDS filter_test benchmark_test options_test basic_test fixture_test cxx03_test complexity_test
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Running LCOV"
    )
    add_custom_target(coverage
      DEPENDS ${CMAKE_BINARY_DIR}/lcov/index.html
      COMMENT "LCOV report at lcov/index.html"
    )
    message(STATUS "Coverage command added")
  else()
    if (HAVE_CXX_FLAG_COVERAGE)
      set(CXX_FLAG_COVERAGE_MESSAGE supported)
    else()
      set(CXX_FLAG_COVERAGE_MESSAGE unavailable)
    endif()
    message(WARNING
      "Coverage not available:\n"
      "  gcov: ${GCOV}\n"
      "  lcov: ${LCOV}\n"
      "  genhtml: ${GENHTML}\n"
      "  ctest: ${CTEST}\n"
      "  --coverage flag: ${CXX_FLAG_COVERAGE_MESSAGE}")
  endif()
endif()
