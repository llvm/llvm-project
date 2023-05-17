if(LIBC_TESTS_CAN_USE_MPFR)
  add_library(libcMPFRWrapper
    MPFRUtils.cpp
    MPFRUtils.h
  )
  add_compile_options(
    -O3
  )
  add_dependencies(
    libcMPFRWrapper 
    libc.src.__support.CPP.string_view 
    libc.src.__support.CPP.type_traits 
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.platform_defs
    LibcUnitTest
  )
  if(EXISTS ${LLVM_LIBC_MPFR_INSTALL_PATH})
    target_include_directories(libcMPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/include)
    target_link_directories(libcMPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/lib)
  endif()
  target_link_libraries(libcMPFRWrapper LibcFPTestHelpers LibcUnitTest mpfr gmp)
elseif(NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
  message(WARNING "Math tests using MPFR will be skipped.")
endif()
