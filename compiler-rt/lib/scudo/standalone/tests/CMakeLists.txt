include_directories(..)

add_custom_target(ScudoUnitTests)
set_target_properties(ScudoUnitTests PROPERTIES
  FOLDER "Compiler-RT Tests")

set(SCUDO_UNITTEST_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${SANITIZER_TEST_CXX_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/scudo/standalone
  -I${COMPILER_RT_SOURCE_DIR}/lib/scudo/standalone/include
  -DGTEST_HAS_RTTI=0
  -g
  # Extra flags for the C++ tests
  # TODO(kostyak): find a way to make -fsized-deallocation work
  -Wno-mismatched-new-delete)

if(COMPILER_RT_DEBUG)
  list(APPEND SCUDO_UNITTEST_CFLAGS -DSCUDO_DEBUG=1)
endif()

if(ANDROID)
  list(APPEND SCUDO_UNITTEST_CFLAGS -fno-emulated-tls)
endif()

if (COMPILER_RT_HAS_GWP_ASAN)
  list(APPEND SCUDO_UNITTEST_CFLAGS -DGWP_ASAN_HOOKS -fno-omit-frame-pointer
       -mno-omit-leaf-frame-pointer)
endif()

set(SCUDO_TEST_ARCH ${SCUDO_STANDALONE_SUPPORTED_ARCH})

# gtests requires c++
set(LINK_FLAGS
  ${COMPILER_RT_UNITTEST_LINK_FLAGS}
  ${SANITIZER_TEST_CXX_LIBRARIES}
  )
list(APPEND LINK_FLAGS -pthread -no-pie)
# Linking against libatomic is required with some compilers
check_library_exists(atomic __atomic_load_8 "" COMPILER_RT_HAS_LIBATOMIC)
if (COMPILER_RT_HAS_LIBATOMIC)
  list(APPEND LINK_FLAGS -latomic)
endif()

set(SCUDO_TEST_HEADERS
  scudo_unit_test.h
  )
foreach (header ${SCUDO_HEADERS})
  list(APPEND SCUDO_TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../${header})
endforeach()

macro(add_scudo_unittest testname)
  cmake_parse_arguments(TEST "" "" "SOURCES;ADDITIONAL_RTOBJECTS" ${ARGN})
  if (COMPILER_RT_HAS_GWP_ASAN)
    list(APPEND TEST_ADDITIONAL_RTOBJECTS
         RTGwpAsan RTGwpAsanBacktraceLibc RTGwpAsanSegvHandler)
  endif()

  if(COMPILER_RT_HAS_SCUDO_STANDALONE)
    foreach(arch ${SCUDO_TEST_ARCH})
      # Additional runtime objects get added along RTScudoStandalone
      set(SCUDO_TEST_RTOBJECTS $<TARGET_OBJECTS:RTScudoStandalone.${arch}>)
      foreach(rtobject ${TEST_ADDITIONAL_RTOBJECTS})
        list(APPEND SCUDO_TEST_RTOBJECTS $<TARGET_OBJECTS:${rtobject}.${arch}>)
      endforeach()
      # Add the static runtime library made of all the runtime objects
      set(RUNTIME RT${testname}.${arch})
      add_library(${RUNTIME} STATIC ${SCUDO_TEST_RTOBJECTS})
      set(ScudoUnitTestsObjects)
      generate_compiler_rt_tests(ScudoUnitTestsObjects ScudoUnitTests
        "${testname}-${arch}-Test" ${arch}
        SOURCES ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
        COMPILE_DEPS ${SCUDO_TEST_HEADERS}
        DEPS llvm_gtest scudo_standalone
        RUNTIME ${RUNTIME}
        CFLAGS ${SCUDO_UNITTEST_CFLAGS}
        LINK_FLAGS ${LINK_FLAGS})
    endforeach()
  endif()
endmacro()

set(SCUDO_UNIT_TEST_SOURCES
  atomic_test.cpp
  bytemap_test.cpp
  checksum_test.cpp
  chunk_test.cpp
  combined_test.cpp
  common_test.cpp
  flags_test.cpp
  list_test.cpp
  map_test.cpp
  memtag_test.cpp
  mutex_test.cpp
  primary_test.cpp
  quarantine_test.cpp
  release_test.cpp
  report_test.cpp
  secondary_test.cpp
  size_class_map_test.cpp
  stats_test.cpp
  strings_test.cpp
  tsd_test.cpp
  vector_test.cpp
  scudo_unit_test_main.cpp
  )

add_scudo_unittest(ScudoUnitTest
  SOURCES ${SCUDO_UNIT_TEST_SOURCES})

set(SCUDO_C_UNIT_TEST_SOURCES
  wrappers_c_test.cpp
  scudo_unit_test_main.cpp
  )

add_scudo_unittest(ScudoCUnitTest
  SOURCES ${SCUDO_C_UNIT_TEST_SOURCES}
  ADDITIONAL_RTOBJECTS RTScudoStandaloneCWrappers)

set(SCUDO_CXX_UNIT_TEST_SOURCES
  wrappers_cpp_test.cpp
  scudo_unit_test_main.cpp
  )

add_scudo_unittest(ScudoCxxUnitTest
  SOURCES ${SCUDO_CXX_UNIT_TEST_SOURCES}
  ADDITIONAL_RTOBJECTS RTScudoStandaloneCWrappers RTScudoStandaloneCxxWrappers)
