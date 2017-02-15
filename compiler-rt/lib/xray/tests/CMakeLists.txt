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

macro(xray_compile obj_list source arch)
  get_filename_component(basename ${source} NAME)
  set(output_obj "${basename}.${arch}.o")
  get_target_flags_for_arch(${arch} TARGET_CFLAGS)
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND COMPILE_DEPS gtest_main xray)
  endif()
  clang_compile(${output_obj} ${source}
    CFLAGS ${XRAY_UNITTEST_CFLAGS} ${TARGET_CFLAGS}
    DEPS ${COMPILE_DEPS})
  list(APPEND ${obj_list} ${output_obj})
endmacro()

macro(add_xray_unittest testname)
  set(XRAY_TEST_ARCH ${XRAY_SUPPORTED_ARCH})
  if (APPLE)
    darwin_filter_host_archs(XRAY_SUPPORTED_ARCH)
  endif()
  if(UNIX)
    foreach(arch ${XRAY_TEST_ARCH})
      cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
      set(TEST_OBJECTS)
      foreach(SOURCE ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE})
        xray_compile(TEST_OBJECTS ${SOURCE} ${arch} ${TEST_HEADERS})
      endforeach()
      get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
      set(TEST_DEPS ${TEST_OBJECTS})
      if(NOT COMPILER_RT_STANDALONE_BUILD)
        list(APPEND TEST_DEPS gtest_main xray)
      endif()
      if(NOT APPLE)
        add_compiler_rt_test(XRayUnitTests ${testname}-${arch}
          OBJECTS ${TEST_OBJECTS}
          DEPS ${TEST_DEPS}
          LINK_FLAGS ${TARGET_LINK_FLAGS}
          -lstdc++ -lm ${CMAKE_THREAD_LIBS_INIT}
          -lpthread
          -L${COMPILER_RT_LIBRARY_OUTPUT_DIR} -lclang_rt.xray-${arch}
          -latomic -ldl -lrt)
      endif()
      # FIXME: Figure out how to run even just the unit tests on APPLE.
    endforeach()
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  add_subdirectory(unit)
endif()
