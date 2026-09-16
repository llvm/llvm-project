include(CheckCXXCompilerFlag)
include(CompilerRTCompile)

set(COPYPROF_UNITTEST_NOINST_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  ${COMPILER_RT_GMOCK_CFLAGS}
  ${SANITIZER_TEST_CXX_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -DSANITIZER_COMMON_NO_REDEFINE_BUILTINS
  -O1
  -g
  -Wno-pedantic
  -fno-omit-frame-pointer
  -fno-rtti)

file(GLOB COPYPROF_HEADERS ../*.h)

set(COPYPROF_NOINST_TEST_SOURCES
  driver.cpp
  copyprof_shadow_test.cpp
  copyprof_state_test.cpp)

include_directories(.. ../..)

set(COPYPROF_UNIT_TEST_HEADERS
  ${COPYPROF_HEADERS})

set(COPYPROF_UNITTEST_LINK_FLAGS
  ${COMPILER_RT_UNITTEST_LINK_FLAGS})
list(APPEND COPYPROF_UNITTEST_LINK_FLAGS -pthread)
list(APPEND COPYPROF_UNITTEST_LINK_FLAGS ${SANITIZER_TEST_CXX_LIBRARIES})
append_list_if(COMPILER_RT_HAS_LIBDL -ldl COPYPROF_UNITTEST_LINK_FLAGS)
append_list_if(COMPILER_RT_HAS_LIBRT -lrt COPYPROF_UNITTEST_LINK_FLAGS)
append_list_if(COMPILER_RT_HAS_LIBM -lm COPYPROF_UNITTEST_LINK_FLAGS)

# Adds CopyProf tests for each architecture.
macro(add_copyprof_tests_for_arch arch)
  set(COPYPROF_TEST_RUNTIME_OBJECTS
    $<TARGET_OBJECTS:RTCopyProf.${arch}>
    $<TARGET_OBJECTS:RTInterception.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonSymbolizerInternal.${arch}>
  )
  set(COPYPROF_TEST_RUNTIME RTCopyProfTest.${arch})
  add_library(${COPYPROF_TEST_RUNTIME} STATIC ${COPYPROF_TEST_RUNTIME_OBJECTS})
  set_target_properties(${COPYPROF_TEST_RUNTIME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT/Tests/Runtime"
  )
  generate_compiler_rt_tests(COPYPROF_NOINST_TEST_OBJECTS
    CopyProfUnitTests "CopyProf-${arch}-UnitTest" ${arch}
    RUNTIME ${COPYPROF_TEST_RUNTIME}
    SOURCES ${COPYPROF_NOINST_TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE} ${COMPILER_RT_GMOCK_SOURCE}
    COMPILE_DEPS ${COPYPROF_UNIT_TEST_HEADERS}
    CFLAGS ${COPYPROF_UNITTEST_NOINST_CFLAGS}
    LINK_FLAGS ${COPYPROF_UNITTEST_LINK_FLAGS})
endmacro()

# CopyProf unit tests testsuite.
add_custom_target(CopyProfUnitTests)
set_target_properties(CopyProfUnitTests PROPERTIES FOLDER "Compiler-RT/Tests")

if(COMPILER_RT_CAN_EXECUTE_TESTS AND COMPILER_RT_DEFAULT_TARGET_ARCH IN_LIST COPYPROF_SUPPORTED_ARCH)
  # CopyProf unit tests are only run on the host machine.
  foreach(arch ${COMPILER_RT_DEFAULT_TARGET_ARCH})
    add_copyprof_tests_for_arch(${arch})
  endforeach()
endif()
