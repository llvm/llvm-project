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
macro(add_xray_unittest testname)
  cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
  if(UNIX AND NOT APPLE)
    foreach(arch ${XRAY_TEST_ARCH})
      set(TEST_OBJECTS)
      generate_compiler_rt_tests(TEST_OBJECTS
        XRayUnitTests "${testname}-${arch}-Test" "${arch}"
        SOURCES ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
        DEPS gtest xray llvm-xray
        CFLAGS ${XRAY_UNITTEST_CFLAGS}
        LINK_FLAGS -fxray-instrument
          ${TARGET_LINK_FLAGS}
          -lstdc++ -lm ${CMAKE_THREAD_LIBS_INIT}
          -lpthread
          -ldl -lrt)
      set_target_properties(XRayUnitTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  add_subdirectory(unit)
endif()
