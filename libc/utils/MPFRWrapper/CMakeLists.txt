if(LIBC_TESTS_CAN_USE_MPFR)
  add_library(libcMPFRWrapper
    MPFRUtils.cpp
    MPFRUtils.h
    mpfr_inc.h
  )
  target_compile_options(libcMPFRWrapper PRIVATE -O3)
  add_dependencies(
    libcMPFRWrapper
    libc.src.__support.CPP.array
    libc.src.__support.CPP.stringstream
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fpbits_str
    LibcTest.unit
  )
  if(EXISTS ${LLVM_LIBC_MPFR_INSTALL_PATH})
    target_include_directories(libcMPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/include)
    target_link_directories(libcMPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/lib)
  endif()
  target_link_libraries(libcMPFRWrapper PUBLIC LibcFPTestHelpers.unit LibcTest.unit mpfr gmp)
elseif(NOT LIBC_TARGET_OS_IS_GPU AND NOT LLVM_LIBC_FULL_BUILD)
  message(WARNING "Math tests using MPFR will be skipped.")
endif()
