include_directories(../rtl)

add_custom_target(TsanUnitTests)
set_target_properties(TsanUnitTests PROPERTIES
  FOLDER "Compiler-RT Tests")

set(TSAN_UNITTEST_CFLAGS
  ${TSAN_CFLAGS}
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/tsan/rtl
  -DGTEST_HAS_RTTI=0)

set(TSAN_TEST_ARCH ${TSAN_SUPPORTED_ARCH})
if(APPLE)

  # Create a static library for test dependencies.
  set(TSAN_TEST_RUNTIME_OBJECTS
    $<TARGET_OBJECTS:RTTsan_dynamic.osx>
    $<TARGET_OBJECTS:RTInterception.osx>
    $<TARGET_OBJECTS:RTSanitizerCommon.osx>
    $<TARGET_OBJECTS:RTSanitizerCommonLibc.osx>
    $<TARGET_OBJECTS:RTUbsan.osx>)
  set(TSAN_TEST_RUNTIME RTTsanTest)
  add_library(${TSAN_TEST_RUNTIME} STATIC ${TSAN_TEST_RUNTIME_OBJECTS})
  set_target_properties(${TSAN_TEST_RUNTIME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  darwin_filter_host_archs(TSAN_SUPPORTED_ARCH TSAN_TEST_ARCH)
  list(APPEND TSAN_UNITTEST_CFLAGS ${DARWIN_osx_CFLAGS})

  set(LINK_FLAGS "-lc++")
  add_weak_symbols("ubsan" LINK_FLAGS)
  add_weak_symbols("sanitizer_common" LINK_FLAGS)
else()
  set(LINK_FLAGS "-fsanitize=thread;-lstdc++;-lm")
endif()

set(TSAN_RTL_HEADERS)
foreach (header ${TSAN_HEADERS})
  list(APPEND TSAN_RTL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../${header})
endforeach()

# add_tsan_unittest(<name>
#                   SOURCES <sources list>
#                   HEADERS <extra headers list>)
macro(add_tsan_unittest testname)
  cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
  if(UNIX)
    foreach(arch ${TSAN_TEST_ARCH})
      set(TsanUnitTestsObjects)
      generate_compiler_rt_tests(TsanUnitTestsObjects TsanUnitTests
        "${testname}-${arch}-Test" ${arch}
        SOURCES ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
        RUNTIME ${TSAN_TEST_RUNTIME}
        COMPILE_DEPS ${TEST_HEADERS} ${TSAN_RTL_HEADERS}
        DEPS gtest tsan
        CFLAGS ${TSAN_UNITTEST_CFLAGS}
        LINK_FLAGS ${LINK_FLAGS})
    endforeach()
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS AND NOT ANDROID)
  add_subdirectory(rtl)
  add_subdirectory(unit)
endif()
