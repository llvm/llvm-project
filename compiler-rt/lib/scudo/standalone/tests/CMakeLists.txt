include_directories(..)

add_custom_target(ScudoUnitTests)
set_target_properties(ScudoUnitTests PROPERTIES
  FOLDER "Compiler-RT Tests")

set(SCUDO_UNITTEST_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/scudo/standalone
  -DGTEST_HAS_RTTI=0)

set(SCUDO_TEST_ARCH ${SCUDO_STANDALONE_SUPPORTED_ARCH})

# gtests requires c++
set(LINK_FLAGS)
foreach(lib ${SANITIZER_TEST_CXX_LIBRARIES})
  list(APPEND LINK_FLAGS -l${lib})
endforeach()
list(APPEND LINK_FLAGS -pthread)

set(TEST_HEADERS)
foreach (header ${SCUDO_HEADERS})
  list(APPEND TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../${header})
endforeach()

# add_scudo_unittest(<name>
#                   SOURCES <sources list>
#                   HEADERS <extra headers list>)
macro(add_scudo_unittest testname)
  cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
  if(COMPILER_RT_HAS_SCUDO_STANDALONE)
    foreach(arch ${SCUDO_TEST_ARCH})
      set(ScudoUnitTestsObjects)
      add_library("RTScudoStandalone.test.${arch}" STATIC
        $<TARGET_OBJECTS:RTScudoStandalone.${arch}>)
      generate_compiler_rt_tests(ScudoUnitTestsObjects ScudoUnitTests
        "${testname}-${arch}-Test" ${arch}
        SOURCES ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
        COMPILE_DEPS ${TEST_HEADERS}
        DEPS gtest scudo_standalone
        RUNTIME RTScudoStandalone.test.${arch}
        CFLAGS ${SCUDO_UNITTEST_CFLAGS}
        LINK_FLAGS ${LINK_FLAGS})
    endforeach()
  endif()
endmacro()

set(SCUDO_UNIT_TEST_SOURCES
  atomic_test.cc
  bytemap_test.cc
  checksum_test.cc
  flags_test.cc
  list_test.cc
  map_test.cc
  mutex_test.cc
  report_test.cc
  secondary_test.cc
  size_class_map_test.cc
  stats_test.cc
  strings_test.cc
  vector_test.cc
  scudo_unit_test_main.cc)

add_scudo_unittest(ScudoUnitTest
  SOURCES ${SCUDO_UNIT_TEST_SOURCES})
