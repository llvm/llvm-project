include(CheckCXXCompilerFlag)
include(CompilerRTCompile)
include(CompilerRTLink)

set(MEMPROF_UNITTEST_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  ${COMPILER_RT_GMOCK_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/lib/
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
  rawprofile.cpp
  driver.cpp)

include_directories(../../../include)

set(MEMPROF_UNIT_TEST_HEADERS
  ${MEMPROF_HEADERS})

if(NOT WIN32)
  list(APPEND MEMPROF_UNITTEST_LINK_FLAGS -pthread)
endif()

set(MEMPROF_UNITTEST_LINK_LIBRARIES ${SANITIZER_TEST_CXX_LIBRARIES})
list(APPEND MEMPROF_UNITTEST_LINK_LIBRARIES "dl")

if(COMPILER_RT_DEFAULT_TARGET_ARCH IN_LIST MEMPROF_SUPPORTED_ARCH)
  # MemProf unit tests are only run on the host machine.
  set(arch ${COMPILER_RT_DEFAULT_TARGET_ARCH})

  add_executable(MemProfUnitTests 
    ${MEMPROF_UNITTESTS}
    ${COMPILER_RT_GTEST_SOURCE}
    ${COMPILER_RT_GMOCK_SOURCE}
    ${MEMPROF_SOURCES}
    $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>)
  set_target_compile_flags(MemProfUnitTests ${MEMPROF_UNITTEST_CFLAGS})
  set_target_link_flags(MemProfUnitTests ${MEMPROF_UNITTEST_LINK_FLAGS})
  target_link_libraries(MemProfUnitTests ${MEMPROF_UNITTEST_LINK_LIBRARIES})

  if (TARGET cxx-headers OR HAVE_LIBCXX)
    add_dependencies(MemProfUnitTests cxx-headers)
  endif()

  set_target_properties(MemProfUnitTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
