if(LIBC_TESTS_CAN_USE_MPFR)
  add_library(libcMPFRWrapper
    MPFRUtils.cpp
    MPFRUtils.h
  )
  add_compile_options(
    -O3
  )
  add_dependencies(libcMPFRWrapper libc.src.__support.CPP.standalone_cpp libc.src.__support.FPUtil.fputil LibcUnitTest)
  if(EXISTS ${LLVM_LIBC_MPFR_INSTALL_PATH})
    target_include_directories(libcMPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/include)
    target_link_directories(libcMPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/lib)
  endif()
  target_link_libraries(libcMPFRWrapper LibcFPTestHelpers LibcUnitTest mpfr gmp)
else()
  message(WARNING "Math tests using MPFR will be skipped.")
endif()
