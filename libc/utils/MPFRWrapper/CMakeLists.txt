if(LIBC_TESTS_CAN_USE_MPFR OR LIBC_TESTS_CAN_USE_MPC)
  add_library(libcMPCommon STATIC
    MPCommon.cpp
    MPCommon.h
    mpfr_inc.h
  )
  _get_common_test_compile_options(compile_options "" "")
  # mpfr/gmp headers do not work with -ffreestanding flag.
  list(REMOVE_ITEM compile_options "-ffreestanding")
  target_compile_options(libcMPCommon PRIVATE -O3 ${compile_options})
  add_dependencies(
    libcMPCommon
    libc.hdr.stdint_proxy
    libc.src.__support.CPP.string
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.FPUtil.bfloat16
    libc.src.__support.FPUtil.cast
    libc.src.__support.FPUtil.fp_bits
  )
  if(EXISTS ${LLVM_LIBC_MPFR_INSTALL_PATH})
    target_include_directories(libcMPCommon PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/include)
    target_link_directories(libcMPCommon PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/lib)
  endif()
  target_include_directories(libcMPCommon PUBLIC ${LIBC_SOURCE_DIR})
  target_link_libraries(libcMPCommon PUBLIC LibcFPTestHelpers.unit mpfr gmp)
elseif(NOT LIBC_TARGET_OS_IS_GPU AND NOT LLVM_LIBC_FULL_BUILD)
  message(WARNING "Math tests using MPFR will be skipped.")
endif()

if(LIBC_TESTS_CAN_USE_MPFR)
  add_library(libcMPFRWrapper STATIC
    MPFRUtils.cpp
    MPFRUtils.h
  )
  _get_common_test_compile_options(compile_options "" "")
  # mpfr/gmp headers do not work with -ffreestanding flag.
  list(REMOVE_ITEM compile_options "-ffreestanding")
  target_compile_options(libcMPFRWrapper PRIVATE ${libc_opt_high_flag} ${compile_options})
  add_dependencies(
    libcMPFRWrapper
    libcMPCommon
    libc.hdr.stdint_proxy
    libc.src.__support.CPP.array
    libc.src.__support.CPP.stringstream
    libc.src.__support.FPUtil.bfloat16
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fpbits_str
    LibcTest.unit
  )
  if(EXISTS ${LLVM_LIBC_MPFR_INSTALL_PATH})
    target_include_directories(libcMPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/include)
    target_link_directories(libcMPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/lib)
  endif()
  target_include_directories(libcMPFRWrapper PUBLIC ${LIBC_SOURCE_DIR})
  target_link_libraries(libcMPFRWrapper PUBLIC libcMPCommon LibcFPTestHelpers.unit LibcTest.unit)
elseif(NOT LIBC_TARGET_OS_IS_GPU AND NOT LLVM_LIBC_FULL_BUILD)
  message(WARNING "Math tests using MPFR will be skipped.")
endif()
