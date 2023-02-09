set(LLVM_LIBC_MPFR_INSTALL_PATH "" CACHE PATH "Path to where MPFR is installed (e.g. C:/src/install or ~/src/install)")

if(LLVM_LIBC_MPFR_INSTALL_PATH)
  set(LIBC_TESTS_CAN_USE_MPFR TRUE)
elseif(LIBC_TARGET_ARCHITECTURE_IS_GPU)
  set(LIBC_TESTS_CAN_USE_MPFR FALSE)
else()
  try_compile(
    LIBC_TESTS_CAN_USE_MPFR
    ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
    ${LIBC_SOURCE_DIR}/utils/MPFRWrapper/check_mpfr.cpp
    LINK_LIBRARIES
      -lmpfr -lgmp
  )
endif()
