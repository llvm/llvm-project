include(CheckCXXCompilerFlag)
include(CompilerRTCompile)
include(CompilerRTLink)

set(MEMPROF_UNITTEST_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  ${COMPILER_RT_GMOCK_CFLAGS}
  ${SANITIZER_TEST_CXX_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/lib/
  -DSANITIZER_COMMON_NO_REDEFINE_BUILTINS
  -O2
  -g
  -fno-rtti
  -Wno-pedantic
  -fno-omit-frame-pointer)

# Suppress warnings for gmock variadic macros for clang and gcc respectively.
append_list_if(SUPPORTS_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS_FLAG -Wno-gnu-zero-variadic-macro-arguments MEMPROF_UNITTEST_CFLAGS)
append_list_if(COMPILER_RT_HAS_WVARIADIC_MACROS_FLAG -Wno-variadic-macros MEMPROF_UNITTEST_CFLAGS)

file(GLOB MEMPROF_HEADERS ../*.h)

set(MEMPROF_SOURCES
  ../memprof_mibmap.cpp
  ../memprof_rawprofile.cpp)

set(MEMPROF_UNITTESTS
  histogram_encoding.cpp
  rawprofile.cpp
  driver.cpp)

include_directories(../../../include)

set(MEMPROF_UNIT_TEST_HEADERS
  ${MEMPROF_HEADERS})

set(MEMPROF_UNITTEST_LINK_FLAGS
  ${COMPILER_RT_UNITTEST_LINK_FLAGS})

if(NOT WIN32)
  list(APPEND MEMPROF_UNITTEST_LINK_FLAGS -pthread)
endif()

if(APPLE)
  list(APPEND MEMPROF_UNITTEST_CFLAGS ${DARWIN_osx_CFLAGS})
  list(APPEND MEMPROF_UNITTEST_LINK_FLAGS ${DARWIN_osx_LINK_FLAGS})

  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)
  list(APPEND MEMPROF_UNITTEST_LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS})
endif()

set(MEMPROF_UNITTEST_DEPS)
if (TARGET cxx-headers OR HAVE_LIBCXX)
  list(APPEND MEMPROF_UNITTEST_DEPS cxx-headers)
endif()

set(MEMPROF_UNITTEST_LINK_LIBRARIES
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${SANITIZER_TEST_CXX_LIBRARIES})
append_list_if(COMPILER_RT_HAS_LIBDL -ldl MEMPROF_UNITTEST_LINK_LIBRARIES)

# Adds memprof tests for each architecture.
macro(add_memprof_tests_for_arch arch)
  if(APPLE)
    set(MEMPROF_TEST_RUNTIME_OBJECTS
      $<TARGET_OBJECTS:RTSanitizerCommon.osx>
      $<TARGET_OBJECTS:RTSanitizerCommonCoverage.osx>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.osx>
      $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.osx>
    )
  else()
    set(MEMPROF_TEST_RUNTIME_OBJECTS
      $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonSymbolizerInternal.${arch}>
    )
  endif()
  set(MEMPROF_TEST_RUNTIME RTMemProfTest.${arch})
  add_library(${MEMPROF_TEST_RUNTIME} STATIC ${MEMPROF_TEST_RUNTIME_OBJECTS})
  set_target_properties(${MEMPROF_TEST_RUNTIME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT/Tests/Runtime")
  set(MEMPROF_TEST_OBJECTS)
  generate_compiler_rt_tests(MEMPROF_TEST_OBJECTS
    MemProfUnitTests "MemProf-${arch}-UnitTest" ${arch}
    RUNTIME ${MEMPROF_TEST_RUNTIME}
    DEPS ${MEMPROF_UNITTEST_DEPS}
    SOURCES ${MEMPROF_UNITTESTS} ${MEMPROF_SOURCES} ${COMPILER_RT_GTEST_SOURCE}
    COMPILE_DEPS ${MEMPROF_UNIT_TEST_HEADERS}
    CFLAGS ${MEMPROF_UNITTEST_CFLAGS}
    LINK_FLAGS ${MEMPROF_UNITTEST_LINK_FLAGS} ${MEMPROF_UNITTEST_LINK_LIBRARIES})
endmacro()

# MemProf unit tests testsuite.
add_custom_target(MemProfUnitTests)
set_target_properties(MemProfUnitTests PROPERTIES FOLDER "Compiler-RT/Tests")
if(COMPILER_RT_CAN_EXECUTE_TESTS AND COMPILER_RT_DEFAULT_TARGET_ARCH IN_LIST MEMPROF_SUPPORTED_ARCH)
  # MemProf unit tests are only run on the host machine.
  foreach(arch ${COMPILER_RT_DEFAULT_TARGET_ARCH})
    add_memprof_tests_for_arch(${arch})
  endforeach()
endif()
