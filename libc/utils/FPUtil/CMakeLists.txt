if(${LIBC_TARGET_MACHINE} MATCHES "^x86.*")
  set(LONG_DOUBLE_HDR LongDoubleBitsX86.h)
else()
  set(LONG_DOUBLE_HDR)
endif()

if(EXISTS ${LIBC_TARGET_MACHINE})
  set(FENV_IMPL ${LIBC_TARGET_MACHINE}/FEnv.h)
else()
  set(FENV_IMPL DummyFEnv.h)
endif()

add_header_library(
  fputil
  HDRS
    ${LONG_DOUBLE_HDR}
    ${FENV_IMPL}
    BasicOperations.h
    BitPatterns.h
    ClassificationFunctions.h
    DivisionAndRemainderOperations.h
    FEnv.h
    FloatOperations.h
    FloatProperties.h
    FPBits.h
    BasicOperations.h
    ManipulationFunctions.h
    NearestIntegerOperations.h
    NormalFloat.h
  DEPENDS
    libc.include.math
    libc.include.errno
    libc.include.fenv
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
