include_directories(..)

add_custom_target(XRayUnitTests)
set_target_properties(XRayUnitTests PROPERTIES FOLDER "XRay unittests")

# Create an XRAY_IMPL_FILES variable which will include all the implementation
# files that are in the lib directory. Unfortunately, when new files are added
# to the implementation, CMake must be run so that this variable is
# re-generated.
file(GLOB XRAY_IMPL_FILES "../*.cc" "../*.h")

set(XRAY_UNITTEST_CFLAGS
  ${XRAY_CFLAGS}
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib/xray
  -I${COMPILER_RT_SOURCE_DIR}/lib)

macro(add_xray_lib library)
  add_library(${library} STATIC ${ARGN})
  set_target_properties(${library} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT Runtime tests")
endmacro()

function(get_xray_lib_for_arch arch lib)
  if(APPLE)
    set(tgt_name "RTXRay.test.osx")
  else()
    set(tgt_name "RTXRay.test.${arch}")
  endif()
  set(${lib} "${tgt_name}" PARENT_SCOPE)
endfunction()

set(XRAY_TEST_ARCH ${XRAY_SUPPORTED_ARCH})
set(XRAY_UNITTEST_LINK_FLAGS ${CMAKE_THREAD_LIBS_INIT})
if (NOT APPLE)
  append_list_if(COMPILER_RT_HAS_LIBM -lm XRAY_UNITTEST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBRT -lrt XRAY_UNITTEST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBDL -ldl XRAY_UNITTEST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBPTHREAD -pthread XRAY_UNITTEST_LINK_FLAGS)
endif()

macro(add_xray_unittest testname)
  cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
  if(UNIX AND NOT APPLE)
    set(CMAKE_DL_LIBS_INIT "")
    foreach(arch ${XRAY_TEST_ARCH})
      set(TEST_OBJECTS)
      get_xray_lib_for_arch(${arch} XRAY_RUNTIME_LIBS)
      generate_compiler_rt_tests(TEST_OBJECTS
        XRayUnitTests "${testname}-${arch}-Test" "${arch}"
        SOURCES ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
        # Note that any change in the implementations will cause all the unit
        # tests to be re-built. This is by design, but may be cumbersome during
        # the build/test cycle.
        COMPILE_DEPS ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
        ${XRAY_HEADERS} ${XRAY_IMPL_FILES}
        RUNTIME "${XRAY_RUNTIME_LIBS}"
        DEPS gtest xray
        CFLAGS ${XRAY_UNITTEST_CFLAGS}
        LINK_FLAGS ${TARGET_LINK_FLAGS} ${XRAY_UNITTEST_LINK_FLAGS} -lstdc++)
      set_target_properties(XRayUnitTests
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  foreach(arch ${XRAY_SUPPORTED_ARCH})
    if (APPLE)
      add_xray_lib("RTXRay.test.osx"
        $<TARGET_OBJECTS:RTXray.osx>
        $<TARGET_OBJECTS:RTXrayFDR.osx>
        $<TARGET_OBJECTS:RTSanitizerCommon.osx>
        $<TARGET_OBJECTS:RTSanitizerCommonLibc.osx>)
    else()
      add_xray_lib("RTXRay.test.${arch}"
        $<TARGET_OBJECTS:RTXray.${arch}>
        $<TARGET_OBJECTS:RTXrayFDR.${arch}>
        $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
        $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>)
    endif()
  endforeach()
  add_subdirectory(unit)
endif()
