add_library(
  LibcUnitTest
  Test.h
  LibcTest.cpp
  LibcTest.h
)
target_include_directories(LibcUnitTest PUBLIC ${LIBC_SOURCE_DIR})
add_dependencies(LibcUnitTest libc.src.__support.CPP.type_traits libc.src.__support.CPP.uint128)
target_link_libraries(LibcUnitTest PUBLIC libc_test_utils)

add_library(
  LibcUnitTestMain
  LibcTestMain.cpp
)

target_include_directories(LibcUnitTestMain PUBLIC ${LIBC_SOURCE_DIR})
add_dependencies(LibcUnitTestMain LibcUnitTest)
target_link_libraries(LibcUnitTestMain PUBLIC LibcUnitTest libc_test_utils)

add_header_library(
  string_utils
  HDRS
    StringUtils.h
  DEPENDS
  libc.src.__support.CPP.type_traits
)

add_library(
  LibcFPTestHelpers
    FPExceptMatcher.cpp
    FPExceptMatcher.h
    FPMatcher.cpp
    FPMatcher.h
)
target_include_directories(LibcFPTestHelpers PUBLIC ${LIBC_SOURCE_DIR})
target_link_libraries(LibcFPTestHelpers LibcUnitTest)
add_dependencies(
  LibcFPTestHelpers
  LibcUnitTest
  libc.utils.UnitTest.string_utils
  libc.src.__support.FPUtil.fputil
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
  libc.src.__support.FPUtil.fputil #FPBits
  libc.src.stdio.printf_core.core_structs
  libc.utils.UnitTest.string_utils
)
