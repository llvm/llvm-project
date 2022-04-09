#TODO(michaelrj): separate the fputil library into individual targets.
add_header_library(
  fputil
  HDRS
    FEnvImpl.h
    BasicOperations.h
    DivisionAndRemainderOperations.h
    FloatProperties.h
    FPBits.h
    BasicOperations.h
    ManipulationFunctions.h
    NearestIntegerOperations.h
    NormalFloat.h
    PlatformDefs.h
    UInt.h
    XFloat.h
  DEPENDS
    libc.include.math
    libc.include.errno
    libc.include.fenv
    libc.src.__support.common
    libc.src.__support.CPP.type_traits
    libc.src.__support.CPP.bit
    libc.src.errno.errno
)

add_header_library(
  sqrt
  HDRS
    sqrt.h
  DEPENDS
    .fputil
    libc.src.__support.FPUtil.generic.sqrt
)

add_header_library(
  fma
  HDRS
    FMA.h
  DEPENDS
    .fputil
    libc.src.__support.FPUtil.generic.fma
)

add_header_library(
  multiply_add
  HDRS
    multiply_add.h
  DEPENDS
    .fma
)

add_header_library(
  polyeval
  HDRS
    PolyEval.h
  DEPENDS
    .multiply_add
)

add_subdirectory(generic)
