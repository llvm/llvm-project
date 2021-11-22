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
  -Wno-gnu-zero-variadic-macro-arguments
  -fno-omit-frame-pointer)

file(GLOB MEMPROF_HEADERS ../*.h)

set(MEMPROF_SOURCES
  ../memprof_mibmap.cpp
  ../memprof_rawprofile.cpp)

set(MEMPROF_UNITTESTS
  rawprofile.cpp
  driver.cpp)

set(MEMPROF_UNIT_TEST_HEADERS
  ${MEMPROF_HEADERS})

if(NOT WIN32)
  list(APPEND MEMPROF_UNITTEST_LINK_FLAGS -pthread)
endif()

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
  target_link_libraries(MemProfUnitTests dl) 

  set_target_properties(MemProfUnitTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
