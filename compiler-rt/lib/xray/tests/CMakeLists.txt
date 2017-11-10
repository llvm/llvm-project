include_directories(..)

add_custom_target(XRayUnitTests)
set_target_properties(XRayUnitTests PROPERTIES FOLDER "XRay unittests")

set(XRAY_UNITTEST_CFLAGS
  ${XRAY_CFLAGS}
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib/xray
  -I${COMPILER_RT_SOURCE_DIR}/lib)

set(XRAY_TEST_ARCH ${XRAY_SUPPORTED_ARCH})
set(XRAY_LINK_FLAGS)
append_list_if(COMPILER_RT_HAS_LIBRT -lrt XRAY_LINK_FLAGS)
append_list_if(COMPILER_RT_HAS_LIBM -lm XRAY_LINK_FLAGS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD -lpthread XRAY_LINK_FLAGS)

if (APPLE)
  list(APPEND XRAY_LINK_FLAGS -lc++)
  list(APPEND XRAY_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})
  set(XRAY_TEST_RUNTIME_OBJECTS
    $<TARGET_OBJECTS:RTSanitizerCommon.osx>
    $<TARGET_OBJECTS:RTSanitizerCommonLibc.osx>
    $<TARGET_OBJECTS:RTXray.osx>)
  set(XRAY_TEST_RUNTIME RTXRayTest)
  add_library(${XRAY_TEST_RUNTIME} STATIC ${XRAY_TEST_RUNTIME_OBJECTS})
  set_target_properties(${XRAY_TEST_RUNTIME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT Runtime tests")

  darwin_filter_host_archs(XRAY_SUPPORTED_ARCH XRAY_TEST_ARCH)
  list(APPEND XRAY_UNITTEST_CFLAGS ${DARWIN_osx_CFLAGS})
  list(APPEND XRAY_LINK_FLAGS "-lc++")
  list(APPEND XRAY_LINK_FLAGS "-fxray-instrument")
  add_weak_symbols("sanitizer_common" XRAY_LINK_FLAGS)
  add_weak_symbols("xray" XRAY_LINK_FLAGS)
else()
  append_list_if(COMPILER_RT_HAS_LIBSTDCXX lstdc++ XRAY_LINK_FLAGS)
endif()

macro(add_xray_unittest testname)
  cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
  if(UNIX)
    foreach(arch ${XRAY_TEST_ARCH})
      set(TEST_OBJECTS)
      generate_compiler_rt_tests(TEST_OBJECTS
        XRayUnitTests "${testname}-${arch}-Test" "${arch}"
        SOURCES ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
        RUNTIME ${XRAY_TEST_RUNTIME}
        COMPILE_DEPS ${TEST_HEADERS}
        DEPS gtest xray llvm-xray
        CFLAGS ${XRAY_UNITTEST_CFLAGS}
        LINK_FLAGS -fxray-instrument
          ${TARGET_LINK_FLAGS}
          ${CMAKE_THREAD_LIBS_INIT}
          ${XRAY_LINK_FLAGS})
      set_target_properties(XRayUnitTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  add_subdirectory(unit)
endif()
