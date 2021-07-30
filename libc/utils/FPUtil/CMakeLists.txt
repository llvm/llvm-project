if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  set(LONG_DOUBLE_HDR LongDoubleBitsX86.h)
else()
  set(LONG_DOUBLE_HDR)
endif()

if(EXISTS ${LIBC_TARGET_ARCHITECTURE})
  set(FENV_IMPL ${LIBC_TARGET_ARCHITECTURE}/FEnvImpl.h)
else()
  set(FENV_IMPL DummyFEnvImpl.h)
endif()

add_header_library(
  fputil
  HDRS
    ${LONG_DOUBLE_HDR}
    ${FENV_IMPL}
    BasicOperations.h
    DivisionAndRemainderOperations.h
    FEnvUtils.h
    FloatProperties.h
    FPBits.h
    BasicOperations.h
    ManipulationFunctions.h
    NearestIntegerOperations.h
    NormalFloat.h
    PlatformDefs.h
    PolyEval.h
  DEPENDS
    libc.include.math
    libc.include.errno
    libc.include.fenv
    libc.src.__support.common
    libc.utils.CPP.standalone_cpp
)

add_library(
  LibcFPTestHelpers
    FPExceptMatcher.cpp
    FPExceptMatcher.h
    TestHelpers.cpp
    TestHelpers.h
)
target_include_directories(LibcFPTestHelpers PUBLIC ${LIBC_SOURCE_DIR})
target_link_libraries(LibcFPTestHelpers LibcUnitTest)
add_dependencies(
  LibcFPTestHelpers
  LibcUnitTest
  libc.utils.CPP.standalone_cpp
  libc.utils.FPUtil.fputil
)
