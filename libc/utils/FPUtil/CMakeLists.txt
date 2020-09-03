if(${LIBC_TARGET_MACHINE} MATCHES "^x86.*")
  set(LONG_DOUBLE_HDR LongDoubleBitsX86.h)
else()
  set(LONG_DOUBLE_HDR)
endif()

add_header_library(
  fputil
  HDRS
    ${LONG_DOUBLE_HDR}
    BitPatterns.h
    FloatOperations.h
    FloatProperties.h
    FPBits.h
    ManipulationFunctions.h
  DEPENDS
    libc.utils.CPP.standalone_cpp
)
