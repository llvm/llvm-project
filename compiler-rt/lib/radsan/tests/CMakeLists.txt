include(CompilerRTCompile)

include_directories(..)

set(RADSAN_UNITTEST_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  ${COMPILER_RT_GMOCK_CFLAGS}
  ${SANITIZER_TEST_CXX_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/lib/
  -I${COMPILER_RT_SOURCE_DIR}/include/
  -I${COMPILER_RT_SOURCE_DIR}/lib/radsan
  -I${COMPILER_RT_SOURCE_DIR}/lib/sanitizer_common/tests
  -DSANITIZER_COMMON_NO_REDEFINE_BUILTINS
  -O2)

set(RADSAN_INST_TEST_SOURCES
    radsan_test.cpp
    radsan_test_interceptors.cpp
    radsan_test_main.cpp)

set(RADSAN_NOINST_TEST_SOURCES
    ../radsan_preinit.cpp
    radsan_test_context.cpp
    radsan_test_main.cpp)

set(RADSAN_UNITTEST_HEADERS
    radsan_test_utilities.h)

add_custom_target(RadsanUnitTests)
set_target_properties(RadsanUnitTests PROPERTIES FOLDER "Compiler-RT Tests")

set(RADSAN_UNITTEST_LINK_FLAGS
  ${COMPILER_RT_UNITTEST_LINK_FLAGS}
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${SANITIZER_TEST_CXX_LIBRARIES}
  -no-pie)

if (APPLE)
  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)
  list(APPEND RADSAN_UNITTEST_LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS})
  list(APPEND RADSAN_UNITTEST_LINK_FLAGS ${DARWIN_osx_LINK_FLAGS})
  list(APPEND RADSAN_UNITTEST_CFLAGS ${DARWIN_osx_CFLAGS})
else()
  list(APPEND RADSAN_UNITTEST_LINK_FLAGS -latomic)
endif()

set(COMPILER_RT_GOOGLETEST_SOURCES ${COMPILER_RT_GTEST_SOURCE} ${COMPILER_RT_GMOCK_SOURCE})

set(RADSAN_TEST_ARCH ${RADSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(RADSAN_SUPPORTED_ARCH RADSAN_TEST_ARCH)
endif()

foreach(arch ${RADSAN_TEST_ARCH})
  set(RadsanTestObjects)
  generate_compiler_rt_tests(RadsanTestObjects
    RadsanUnitTests "Radsan-${arch}-Test" ${arch}
    COMPILE_DEPS ${RADSAN_UNITTEST_HEADERS}
    SOURCES ${RADSAN_INST_TEST_SOURCES} ${COMPILER_RT_GOOGLETEST_SOURCES}
    DEPS llvm_gtest radsan
    CFLAGS ${RADSAN_UNITTEST_CFLAGS} -fsanitize=realtime
    LINK_FLAGS ${RADSAN_UNITTEST_LINK_FLAGS} -fsanitize=realtime)
  set_target_properties(RadsanUnitTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set(RADSAN_TEST_RUNTIME RTRadsanTest.${arch})
  if(APPLE)
    set(RADSAN_TEST_RUNTIME_OBJECTS
      $<TARGET_OBJECTS:RTRadsan.osx>
      $<TARGET_OBJECTS:RTInterception.osx>
      $<TARGET_OBJECTS:RTSanitizerCommon.osx>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.osx>
      $<TARGET_OBJECTS:RTSanitizerCommonCoverage.osx>
      $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.osx>)
  else()
    set(RADSAN_TEST_RUNTIME_OBJECTS
      $<TARGET_OBJECTS:RTRadsan.${arch}>
      $<TARGET_OBJECTS:RTInterception.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonSymbolizerInternal.${arch}>)
  endif()
  add_library(${RADSAN_TEST_RUNTIME} STATIC ${RADSAN_TEST_RUNTIME_OBJECTS})
  set_target_properties(${RADSAN_TEST_RUNTIME} PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      FOLDER "Compiler-RT Runtime tests")

  set(RadsanNoInstTestObjects)
  generate_compiler_rt_tests(RadsanNoInstTestObjects
    RadsanUnitTests "Radsan-${arch}-NoInstTest" ${arch}
    COMPILE_DEPS ${RADSAN_UNITTEST_HEADERS}
    SOURCES ${RADSAN_NOINST_TEST_SOURCES}
            ${COMPILER_RT_GOOGLETEST_SOURCES}
    DEPS llvm_gtest
    CFLAGS ${RADSAN_UNITTEST_CFLAGS}
    LINK_FLAGS ${RADSAN_UNITTEST_LINK_FLAGS}
    RUNTIME ${RADSAN_TEST_RUNTIME})
  set_target_properties(RadsanUnitTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
