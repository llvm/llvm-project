include(CheckCXXCompilerFlag)
include(CompilerRTCompile)
include(CompilerRTLink)

set(CTX_PROFILE_UNITTEST_CFLAGS
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
append_list_if(SUPPORTS_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS_FLAG -Wno-gnu-zero-variadic-macro-arguments CTX_PROFILE_UNITTEST_CFLAGS)
append_list_if(COMPILER_RT_HAS_WVARIADIC_MACROS_FLAG -Wno-variadic-macros CTX_PROFILE_UNITTEST_CFLAGS)

file(GLOB PROFILE_HEADERS ../*.h)

set(CTX_PROFILE_SOURCES
  ../CtxInstrProfiling.cpp)

set(CTX_PROFILE_UNITTESTS
  CtxInstrProfilingTest.cpp
  driver.cpp)

include_directories(../../../include)

set(CTX_PROFILE_UNIT_TEST_HEADERS
  ${CTX_PROFILE_HEADERS})

set(CTX_PROFILE_UNITTEST_LINK_FLAGS
  ${COMPILER_RT_UNITTEST_LINK_FLAGS})

list(APPEND CTX_PROFILE_UNITTEST_LINK_FLAGS -pthread)

set(CTX_PROFILE_UNITTEST_LINK_LIBRARIES
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${SANITIZER_TEST_CXX_LIBRARIES})
list(APPEND CTX_PROFILE_UNITTEST_LINK_LIBRARIES "dl")

if(COMPILER_RT_DEFAULT_TARGET_ARCH IN_LIST CTX_PROFILE_SUPPORTED_ARCH)
  # Profile unit tests are only run on the host machine.
  set(arch ${COMPILER_RT_DEFAULT_TARGET_ARCH})

  add_executable(CtxProfileUnitTests 
    ${CTX_PROFILE_UNITTESTS}
    ${COMPILER_RT_GTEST_SOURCE}
    ${COMPILER_RT_GMOCK_SOURCE}
    ${CTX_PROFILE_SOURCES}
    $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonSymbolizerInternal.${arch}>)
  set_target_compile_flags(CtxProfileUnitTests ${CTX_PROFILE_UNITTEST_CFLAGS})
  set_target_link_flags(CtxProfileUnitTests ${CTX_PROFILE_UNITTEST_LINK_FLAGS})
  target_link_libraries(CtxProfileUnitTests ${CTX_PROFILE_UNITTEST_LINK_LIBRARIES})

  if (TARGET cxx-headers OR HAVE_LIBCXX)
    add_dependencies(CtxProfileUnitTests cxx-headers)
  endif()

  set_target_properties(CtxProfileUnitTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
