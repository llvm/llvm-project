include_directories(..)

add_custom_target(XRayUnitTests)
set_target_properties(XRayUnitTests PROPERTIES FOLDER "XRay unittests")

# Create an XRAY_IMPL_FILES variable which must include all the implementation
# files that are in the lib directory. We *must* keep this list up-to-date.
set(XRAY_IMPL_FILES
  ../../xray_AArch64.cc
  ../../xray_allocator.h
  ../../xray_arm.cc
  ../../xray_basic_flags.cc
  ../../xray_basic_flags.h
  ../../xray_basic_logging.cc
  ../../xray_basic_logging.h
  ../../xray_buffer_queue.cc
  ../../xray_buffer_queue.h
  ../../xray_defs.h
  ../../xray_fdr_flags.cc
  ../../xray_fdr_flags.h
  ../../xray_fdr_logging.cc
  ../../xray_fdr_logging.h
  ../../xray_fdr_log_records.h
  ../../xray_flags.cc
  ../../xray_flags.h
  ../../xray_function_call_trie.h
  ../../xray_init.cc
  ../../xray_interface.cc
  ../../xray_interface_internal.h
  ../../xray_log_interface.cc
  ../../xray_mips64.cc
  ../../xray_mips.cc
  ../../xray_powerpc64.cc
  ../../xray_profile_collector.cc
  ../../xray_profile_collector.h
  ../../xray_profiler_flags.cc
  ../../xray_profiler_flags.h
  ../../xray_segmented_array.h
  ../../xray_trampoline_powerpc64.cc
  ../../xray_tsc.h
  ../../xray_utils.cc
  ../../xray_utils.h
  ../../xray_x86_64.cc)

set(XRAY_UNITTEST_CFLAGS
  ${XRAY_CFLAGS}
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib/xray
  -I${COMPILER_RT_SOURCE_DIR}/lib)

function(add_xray_lib library)
  add_library(${library} STATIC ${ARGN})
  set_target_properties(${library} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT Runtime tests")
endfunction()

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
  if (APPLE)
    add_xray_lib("RTXRay.test.osx"
      $<TARGET_OBJECTS:RTXray.osx>
      $<TARGET_OBJECTS:RTXrayFDR.osx>
      $<TARGET_OBJECTS:RTXrayPROFILER.osx>
      $<TARGET_OBJECTS:RTSanitizerCommon.osx>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.osx>)
  else()
  foreach(arch ${XRAY_SUPPORTED_ARCH})
    add_xray_lib("RTXRay.test.${arch}"
      $<TARGET_OBJECTS:RTXray.${arch}>
      $<TARGET_OBJECTS:RTXrayFDR.${arch}>
      $<TARGET_OBJECTS:RTXrayPROFILER.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>)
  endforeach()
  endif()
  add_subdirectory(unit)
endif()
