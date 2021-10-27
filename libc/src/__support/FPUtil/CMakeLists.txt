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
    UInt.h
    XFloat.h
  DEPENDS
    libc.include.math
    libc.include.errno
    libc.include.fenv
    libc.src.__support.common
    libc.src.__support.CPP.standalone_cpp
)
