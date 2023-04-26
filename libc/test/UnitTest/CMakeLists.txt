set(libc_test_srcs_common
  Test.h
  LibcTest.cpp
  CmakeFilePath.cpp
  LibcTest.h
  TestLogger.cpp
  TestLogger.h
)

set(libc_death_test_srcs ExecuteFunction.h)
if(${LIBC_TARGET_OS} STREQUAL "linux")
  list(APPEND libc_death_test_srcs 
       LibcDeathTestExecutors.cpp ExecuteFunctionUnix.cpp)
endif()

add_library(
  LibcUnitTest
  ${libc_test_srcs_common}
  ${libc_death_test_srcs}
)

add_library(
  LibcHermeticTest
  ${libc_test_srcs_common}
  HermeticTestUtils.cpp
)

foreach(lib LibcUnitTest LibcHermeticTest)
  target_include_directories(${lib} PUBLIC ${LIBC_SOURCE_DIR})
  if(LIBC_GPU_BUILD OR LIBC_GPU_ARCHITECTURES)
    # If we are in full build mode, then we should use our own public headers.
    target_include_directories(${lib} PRIVATE ${LIBC_BUILD_DIR}/include)
  endif()
  target_compile_options(${lib} PRIVATE ${LIBC_HERMETIC_TEST_COMPILE_OPTIONS}
                         -fno-exceptions -fno-rtti)
  add_dependencies(${lib}
    libc.src.__support.c_string
    libc.src.__support.CPP.string
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.OSUtil.osutil
    libc.src.__support.uint128
  )
endforeach()

target_compile_options(LibcHermeticTest PRIVATE -ffreestanding -nostdlib -nostdlib++)

add_library(
  LibcTestMain
  LibcTestMain.cpp
)

target_include_directories(LibcTestMain PUBLIC ${LIBC_SOURCE_DIR})
target_compile_options(LibcTestMain PRIVATE -fno-exceptions -fno-rtti)
if(LIBC_GPU_BUILD OR LIBC_GPU_ARCHITECTURES)
  target_include_directories(LibcTestMain PRIVATE ${LIBC_BUILD_DIR}/include)
  target_compile_options(LibcTestMain PRIVATE ${LIBC_HERMETIC_TEST_COMPILE_OPTIONS})
endif()
add_dependencies(LibcTestMain LibcUnitTest)

add_header_library(
  string_utils
  HDRS
  StringUtils.h
  DEPENDS
    libc.src.__support.CPP.string
    libc.src.__support.CPP.type_traits
)

add_library(
  LibcFPTestHelpers
  FPExceptMatcher.cpp
  FPExceptMatcher.h
  FPMatcher.h
  RoundingModeUtils.cpp
  RoundingModeUtils.h
)
target_include_directories(LibcFPTestHelpers PUBLIC ${LIBC_SOURCE_DIR})
target_link_libraries(LibcFPTestHelpers LibcUnitTest)
add_dependencies(
  LibcFPTestHelpers
  LibcUnitTest
  libc.test.UnitTest.string_utils
  libc.src.__support.FPUtil.fp_bits
  libc.src.__support.FPUtil.fenv_impl
  libc.test.UnitTest.string_utils
)

add_library(
  LibcMemoryHelpers
  MemoryMatcher.h
  MemoryMatcher.cpp
)
target_include_directories(LibcMemoryHelpers PUBLIC ${LIBC_SOURCE_DIR})
target_link_libraries(LibcMemoryHelpers LibcUnitTest)
add_dependencies(
  LibcMemoryHelpers
  LibcUnitTest
  libc.src.__support.CPP.span
)

add_library(
  LibcPrintfHelpers
  PrintfMatcher.h
  PrintfMatcher.cpp
)
target_include_directories(LibcPrintfHelpers PUBLIC ${LIBC_SOURCE_DIR})
target_link_libraries(LibcPrintfHelpers LibcUnitTest)
add_dependencies(
  LibcPrintfHelpers
  LibcUnitTest
  libc.src.__support.FPUtil.fp_bits
  libc.src.stdio.printf_core.core_structs
  libc.test.UnitTest.string_utils
)

add_library(
  LibcScanfHelpers
  ScanfMatcher.h
  ScanfMatcher.cpp
)
target_include_directories(LibcScanfHelpers PUBLIC ${LIBC_SOURCE_DIR})
target_link_libraries(LibcScanfHelpers LibcUnitTest)
add_dependencies(
  LibcScanfHelpers
  LibcUnitTest
  libc.src.__support.FPUtil.fp_bits
  libc.src.stdio.scanf_core.core_structs
  libc.test.UnitTest.string_utils
)
