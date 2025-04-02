if(LIBC_TESTS_CAN_USE_MPC)
  add_library(libcMPCWrapper STATIC
    MPCUtils.cpp
    MPCUtils.h
  )
  _get_common_test_compile_options(compile_options "" "")
  list(REMOVE_ITEM compile_options "-ffreestanding")
  target_compile_options(libcMPCWrapper PRIVATE -O3 ${compile_options})
  add_dependencies(
    libcMPCWrapper
    libcMPCommon
    libc.src.__support.CPP.array
    libc.src.__support.CPP.string
    libc.src.__support.CPP.stringstream
    libc.src.__support.CPP.type_traits
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.complex_type
    LibcTest.unit
  )
  if(EXISTS ${LLVM_LIBC_MPFR_INSTALL_PATH})
    target_include_directories(libcMPCWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/include)
    target_link_directories(libcMPCWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/lib)
  endif()
  if(EXISTS ${LIBC_MPC_INSTALL_PATH})
    target_include_directories(libcMPCWrapper PUBLIC ${LIBC_MPC_INSTALL_PATH}/include)
    target_link_directories(libcMPCWrapper PUBLIC ${LIBC_MPC_INSTALL_PATH}/lib)
  endif()
  target_include_directories(libcMPCWrapper PUBLIC ${LIBC_SOURCE_DIR})
  target_link_libraries(libcMPCWrapper PUBLIC libcMPCommon LibcFPTestHelpers.unit LibcTest.unit mpc)
elseif(NOT LIBC_TARGET_OS_IS_GPU AND NOT LLVM_LIBC_FULL_BUILD)
  message(WARNING "Math tests using MPC will be skipped.")
endif()
