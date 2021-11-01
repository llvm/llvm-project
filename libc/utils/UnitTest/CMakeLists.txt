add_library(
  LibcUnitTest
  Test.h
  LibcTest.cpp
  LibcTest.h
)
target_include_directories(LibcUnitTest PUBLIC ${LIBC_SOURCE_DIR})
add_dependencies(LibcUnitTest libc.src.__support.CPP.standalone_cpp)
target_link_libraries(LibcUnitTest PUBLIC libc_test_utils)

add_library(
  LibcUnitTestMain
  LibcTestMain.cpp
)

target_include_directories(LibcUnitTestMain PUBLIC ${LIBC_SOURCE_DIR})
add_dependencies(LibcUnitTestMain LibcUnitTest libc.src.__support.CPP.standalone_cpp)
target_link_libraries(LibcUnitTestMain PUBLIC LibcUnitTest libc_test_utils)

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
  libc.src.__support.CPP.standalone_cpp
  libc.src.__support.FPUtil.fputil
)
