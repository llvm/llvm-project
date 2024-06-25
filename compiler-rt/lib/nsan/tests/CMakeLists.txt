include(CompilerRTCompile)

set(NSAN_UNITTEST_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  ${SANITIZER_TEST_CXX_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -DSANITIZER_COMMON_REDEFINE_BUILTINS_IN_STD
  -O2
  -g
  -fno-omit-frame-pointer)

set(NSAN_UNITTEST_LINK_FLAGS
  ${COMPILER_RT_UNITTEST_LINK_FLAGS}
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${SANITIZER_TEST_CXX_LIBRARIES})

file(GLOB NSAN_HEADERS ../*.h)
set(NSAN_UNITTESTS
	NSanUnitTest.cpp
        nsan_unit_test_main.cpp)

add_custom_target(NsanUnitTests)

# set(NSAN_UNITTEST_LINK_FLAGS ${COMPILER_RT_UNITTEST_LINK_FLAGS} -ldl)
# list(APPEND NSAN_UNITTEST_LINK_FLAGS --driver-mode=g++)

if(COMPILER_RT_DEFAULT_TARGET_ARCH IN_LIST NSAN_SUPPORTED_ARCH)
  # NSan unit tests are only run on the host machine.
  set(arch ${COMPILER_RT_DEFAULT_TARGET_ARCH})

  set(NSAN_TEST_RUNTIME RTNsanTest.${arch})

  set(NSAN_TEST_RUNTIME_OBJECTS
    $<TARGET_OBJECTS:RTNsan.${arch}>
    $<TARGET_OBJECTS:RTInterception.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>)

  add_library(${NSAN_TEST_RUNTIME} STATIC
    ${NSAN_TEST_RUNTIME_OBJECTS})

  set_target_properties(${NSAN_TEST_RUNTIME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT Runtime tests")

  set(NsanTestObjects)
  generate_compiler_rt_tests(NsanTestObjects
    NsanUnitTests "Nsan-${arch}-Test" ${arch}
    SOURCES ${NSAN_UNITTESTS} ${COMPILER_RT_GTEST_SOURCE}
    RUNTIME ${NSAN_TEST_RUNTIME}
    DEPS ${NSAN_UNIT_TEST_HEADERS}
    CFLAGS ${NSAN_UNITTEST_CFLAGS}
    LINK_FLAGS ${NSAN_UNITTEST_LINK_FLAGS})
  set_target_properties(NsanUnitTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

