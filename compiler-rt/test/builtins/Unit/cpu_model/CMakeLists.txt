include(CheckCXXCompilerFlag)
include(CompilerRTCompile)
include(CompilerRTLink)

set(BUILTINS_CPUMODEL_UNITTEST_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  ${COMPILER_RT_GMOCK_CFLAGS}
  ${SANITIZER_TEST_CXX_CFLAGS}
  -fno-builtin
  -I${COMPILER_RT_SOURCE_DIR}/lib/builtins
  -I${CMAKE_CURRENT_SOURCE_DIR}
  -nodefaultlibs)

set(BUILTINS_CPUMODEL_UNITTEST_DEPS)
if (TARGET cxx-headers OR HAVE_LIBCXX)
  list(APPEND BUILTINS_CPUMODEL_UNITTEST_DEPS cxx-headers)
endif()

set(BUILTINS_CPUMODEL_UNITTESTS
  x86.cpp)

set(BUILTINS_CPUMODEL_SOURCES
  ../../../../lib/builtins/cpu_model/x86.c
  cpuid.cpp)

set(BUILTINS_CPUMODEL_UNITTEST_LINK_LIBRARIES
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${SANITIZER_TEST_CXX_LIBRARIES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(BUILTINS_CPUMODEL_UNITTEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cpuid.h)

macro (add_builtins_cpumodel_tests_for_arch arch)
  set(BUILTINS_CPUMODEL_TEST_RUNTIME_OBJECTS
    $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonSymbolizerInternal.${arch}>
  )
  set(BUILTINS_CPUMODEL_TEST_RUNTIME RTBuiltinsCPUModelTest.${arch})
  add_library(${BUILTINS_CPUMODEL_TEST_RUNTIME} STATIC ${BUILTINS_CPUMODEL_TEST_RUNTIME_OBJECTS})
  set_target_properties(${BUILTINS_CPUMODEL_TEST_RUNTIME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER "Compiler-RT Runtime tests")
  set(BUILTSIN_CPUMODEL_TEST_OBJECTS)
  generate_compiler_rt_tests(BUILTINS_CPUMODEL_TEST_OBJECTS
    BuiltinsCPUModelUnitTests "BuiltinsCPUModel-${arch}-UnitTest" ${arch}
    RUNTIME ${BUILTINS_CPUMODEL_TEST_RUNTIME}
    DEPS ${BUILTINS_CPUMODEL_UNITTEST_DEPS}
    SOURCES ${BUILTINS_CPUMODEL_SOURCES} ${BUILTINS_CPUMODEL_UNITTESTS} ${COMPILER_RT_GTEST_SOURCE}
    CFLAGS ${BUILTINS_CPUMODEL_UNITTEST_CFLAGS}
    COMPILE_DEPS ${BUILTINS_CPUMODEL_UNITTEST_HEADERS}
    LINK_FLAGS ${COMPILER_RT_UNITTEST_LINK_FLAGS} ${BUILTINS_CPUMODEL_UNITTEST_LINK_LIBRARIES})
endmacro()

add_custom_target(BuiltinsCPUModelUnitTests)
set_target_properties(BuiltinsCPUModelUnitTests PROPERTIES FOLDER "Compiler-RT Tests")

set(BUILTIN_TEST_ARCH ${BUILTIN_SUPPORTED_ARCH})

# Only add the tests on x86_64 instead of looping over all the arches,
# because that is where they are supported.
if (COMPILER_RT_CAN_EXECUTE_TESTS AND "x86_64" IN_LIST COMPILER_RT_DEFAULT_TARGET_ARCH)
  set(arch "x86_64")

  add_builtins_cpumodel_tests_for_arch(${arch})
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)
endif()
