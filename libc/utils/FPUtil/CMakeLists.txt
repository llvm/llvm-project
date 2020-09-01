if(${LIBC_TARGET_MACHINE} MATCHES "^x86.*")
  set(LONG_DOUBLE_HDR LongDoubleBitsX86.h)
else()
  set(LONG_DOUBLE_HDR)
endif()

add_header_library(
  fputil
  HDRS
    ${LONG_DOUBLE_HDR}
    BasicOperations.h
    BitPatterns.h
    ClassificationFunctions.h
    FloatOperations.h
    FloatProperties.h
    FPBits.h
    BasicOperations.h
    ManipulationFunctions.h
    NearestIntegerOperations.h
    NormalFloat.h
  DEPENDS
    libc.utils.CPP.standalone_cpp
)

add_llvm_library(
  LibcFPTestHelpers
    TestHelpers.cpp
    TestHelpers.h
)
target_include_directories(LibcFPTestHelpers PUBLIC ${LIBC_SOURCE_DIR})
target_link_libraries(LibcFPTestHelpers LibcUnitTest LLVMSupport)
add_dependencies(
  LibcFPTestHelpers
  LibcUnitTest
  libc.utils.CPP.standalone_cpp
  libc.utils.FPUtil.fputil
)
