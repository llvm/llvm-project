set(LIBFUZZER_UNITTEST_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/lib/fuzzer
  -fno-rtti
  -Werror
  -O2)

add_custom_target(FuzzerUnitTests)
set_target_properties(FuzzerUnitTests PROPERTIES FOLDER "Compiler-RT Tests")

set(LIBFUZZER_UNITTEST_LINK_FLAGS ${COMPILER_RT_UNITTEST_LINK_FLAGS})
list(APPEND LIBFUZZER_UNITTEST_LINK_FLAGS --driver-mode=g++)

if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
  list(APPEND LIBFUZZER_UNITTEST_LINK_FLAGS -lc++ -lpthread)
else()
  list(APPEND LIBFUZZER_UNITTEST_LINK_FLAGS -lstdc++ -lpthread)
endif()

foreach(arch ${FUZZER_SUPPORTED_ARCH})
  set(LIBFUZZER_TEST_RUNTIME RTFuzzerTest.${arch})
  if(APPLE)
    set(LIBFUZZER_TEST_RUNTIME_OBJECTS
      $<TARGET_OBJECTS:RTfuzzer.osx>)
  else()
    set(LIBFUZZER_TEST_RUNTIME_OBJECTS
      $<TARGET_OBJECTS:RTfuzzer.${arch}>)
  endif()
  add_library(${LIBFUZZER_TEST_RUNTIME} STATIC
    ${LIBFUZZER_TEST_RUNTIME_OBJECTS})
  set_target_properties(${LIBFUZZER_TEST_RUNTIME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT Runtime tests")

  set(FuzzerTestObjects)
  generate_compiler_rt_tests(FuzzerTestObjects
    FuzzerUnitTests "Fuzzer-${arch}-Test" ${arch}
    SOURCES FuzzerUnittest.cpp ${COMPILER_RT_GTEST_SOURCE}
    RUNTIME ${LIBFUZZER_TEST_RUNTIME}
    DEPS gtest
    CFLAGS ${LIBFUZZER_UNITTEST_CFLAGS}
    LINK_FLAGS ${LIBFUZZER_UNITTEST_LINK_FLAGS})
  set_target_properties(FuzzerUnitTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
