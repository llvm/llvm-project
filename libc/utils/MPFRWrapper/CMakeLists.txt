try_compile(
  LIBC_TESTS_CAN_USE_MPFR
  ${CMAKE_CURRENT_BINARY_DIR}
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/check_mpfr.cpp
  LINK_LIBRARIES
    -lmpfr -lgmp
)

if(LIBC_TESTS_CAN_USE_MPFR)
  add_library(libcMPFRWrapper
    MPFRUtils.cpp
    MPFRUtils.h
  )
else()
  message(WARNING "Math tests using MPFR will be skipped.")
endif()
