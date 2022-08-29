include(CompilerRTCompile)

include_directories(..)

add_custom_target(OrcRTUnitTests)
set_target_properties(OrcRTUnitTests PROPERTIES FOLDER "OrcRT unittests")

set(ORC_UNITTEST_CFLAGS
  ${ORC_CFLAGS}
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/lib/orc
  )

function(add_orc_lib library)
  add_library(${library} STATIC ${ARGN})
  set_target_properties(${library} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT Runtime tests")
endfunction()

function(get_orc_lib_for_arch arch lib)
  if(APPLE)
    set(tgt_name "RTOrc.test.osx")
  else()
    set(tgt_name "RTOrc.test.${arch}")
  endif()
  set(${lib} "${tgt_name}" PARENT_SCOPE)
endfunction()

set(ORC_TEST_ARCH ${ORC_SUPPORTED_ARCH})
set(ORC_UNITTEST_LINK_FLAGS
  ${COMPILER_RT_UNITTEST_LINK_FLAGS}
  ${CMAKE_THREAD_LIBS_INIT}
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${COMPILER_RT_CXX_LINK_LIBS})

if(APPLE)
  darwin_filter_host_archs(ORC_SUPPORTED_ARCH ORC_TEST_ARCH)
  list(APPEND ORC_UNITTEST_CFLAGS ${DARWIN_osx_CFLAGS})
  list(APPEND ORC_UNITTEST_LINK_FLAGS ${DARWIN_osx_LINK_FLAGS})
else()
  append_list_if(COMPILER_RT_HAS_LIBM -lm ORC_UNITTEST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBRT -lrt ORC_UNITTEST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBDL -ldl ORC_UNITTEST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBPTHREAD -pthread ORC_UNITTEST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBEXECINFO -lexecinfo ORC_UNITTEST_LINK_FLAGS)
endif()

set(ORC_DEPS llvm_gtest orc)
# ORC uses C++ standard library headers.
if (TARGET cxx-headers OR HAVE_LIBCXX)
  set(ORC_DEPS cxx-headers)
endif()

macro(add_orc_unittest testname)
  cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
  if(UNIX)
    foreach(arch ${ORC_TEST_ARCH})
      set(TEST_OBJECTS)
      get_orc_lib_for_arch(${arch} ORC_RUNTIME_LIBS)
      generate_compiler_rt_tests(TEST_OBJECTS
        OrcRTUnitTests "${testname}-${arch}-Test" "${arch}"
        SOURCES ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
        RUNTIME "${ORC_RUNTIME_LIBS}"
        COMPILE_DEPS ${TEST_HEADERS} ${ORC_HEADERS}
        DEPS ${ORC_DEPS}
        CFLAGS ${ORC_UNITTEST_CFLAGS}
        LINK_FLAGS ${ORC_UNITTEST_LINK_FLAGS})
    endforeach()
  endif()
endmacro()

set(UNITTEST_SOURCES
  adt_test.cpp
  c_api_test.cpp
  endian_test.cpp
  error_test.cpp
  executor_address_test.cpp
  extensible_rtti_test.cpp
  orc_unit_test_main.cpp
  wrapper_function_utils_test.cpp
  simple_packed_serialization_test.cpp
  )

if (COMPILER_RT_CAN_EXECUTE_TESTS)

  if (APPLE)
    add_orc_lib("RTOrc.test.osx"
      $<TARGET_OBJECTS:RTOrc.osx>)
  else()
    foreach(arch ${ORC_SUPPORTED_ARCH})
      add_orc_lib("RTOrc.test.${arch}"
        $<TARGET_OBJECTS:RTOrc.${arch}>)
    endforeach()
  endif()

  add_orc_unittest(OrcUnitTest SOURCES ${UNITTEST_SOURCES})

endif()
