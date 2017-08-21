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

macro(add_xray_unittest testname)
  set(XRAY_TEST_ARCH ${XRAY_SUPPORTED_ARCH})
  cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
  # FIXME: Figure out how to run even just the unit tests on APPLE.
  if(UNIX AND NOT APPLE)
    foreach(arch ${XRAY_TEST_ARCH})
      set(TEST_OBJECTS)
      generate_compiler_rt_tests(TEST_OBJECTS
        XRayUnitTests "${testname}-${arch}" "${arch}"
        SOURCES ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
        DEPS gtest_main xray
        CFLAGS ${XRAY_UNITTEST_CFLAGS}
        LINK_FLAGS
          -lstdc++ -lm ${CMAKE_THREAD_LIBS_INIT}
          -lpthread
          -L${COMPILER_RT_LIBRARY_OUTPUT_DIR} -lclang_rt.xray-${arch}
          -ldl -lrt)
    endforeach()
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  add_subdirectory(unit)
endif()
