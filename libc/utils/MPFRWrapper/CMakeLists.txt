if(LIBC_TESTS_CAN_USE_MPFR OR LIBC_TESTS_CAN_USE_MPC)
  add_library(libc.utils.MPCommon STATIC
    MPCommon.cpp
    MPCommon.h
    mpfr_inc.h
  )
  _get_common_test_compile_options(compile_options "" "")
  # mpfr/gmp headers do not work with -ffreestanding flag.
  list(REMOVE_ITEM compile_options "-ffreestanding")
  target_compile_options(libc.utils.MPCommon PRIVATE ${compile_options})
  add_dependencies(
    libc.utils.MPCommon
    libc.hdr.stdint_proxy
    libc.src.__support.CPP.string
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.FPUtil.bfloat16
    libc.src.__support.FPUtil.cast
    libc.src.__support.FPUtil.float80
    libc.src.__support.FPUtil.fp_bits
    libc.test.UnitTest.FPTestHelpers
  )
  if(EXISTS ${LLVM_LIBC_MPFR_INSTALL_PATH})
    target_include_directories(libc.utils.MPCommon PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/include)
    target_link_directories(libc.utils.MPCommon PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/lib)
  endif()
  target_include_directories(libc.utils.MPCommon PUBLIC ${LIBC_SOURCE_DIR})
  target_link_libraries(libc.utils.MPCommon PUBLIC mpfr gmp)
  if(NOT LIBC_TARGET_OS_IS_DARWIN AND NOT LIBC_TARGET_OS_IS_FREEBSD)
    target_link_libraries(libc.utils.MPCommon PUBLIC -latomic)
  endif()
elseif(NOT LIBC_TARGET_OS_IS_GPU AND NOT LLVM_LIBC_FULL_BUILD)
  message(WARNING "Math tests using MPFR will be skipped.")
endif()

if(LIBC_TESTS_CAN_USE_MPFR)
  add_library(libc.utils.MPFRWrapper STATIC
    MPFRUtils.cpp
    MPFRUtils.h
  )
  _get_common_test_compile_options(compile_options "" "")
  # mpfr/gmp headers do not work with -ffreestanding flag.
  list(REMOVE_ITEM compile_options "-ffreestanding")
  target_compile_options(libc.utils.MPFRWrapper PRIVATE ${compile_options})
  add_dependencies(
    libc.utils.MPFRWrapper
    libc.utils.MPCommon
    libc.hdr.stdint_proxy
    libc.src.__support.CPP.array
    libc.src.__support.CPP.stringstream
    libc.src.__support.FPUtil.bfloat16
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fpbits_str
    libc.test.UnitTest.FPTestHelpers
    libc.test.UnitTest.LibcTest
  )
  if(EXISTS ${LLVM_LIBC_MPFR_INSTALL_PATH})
    target_include_directories(libc.utils.MPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/include)
    target_link_directories(libc.utils.MPFRWrapper PUBLIC ${LLVM_LIBC_MPFR_INSTALL_PATH}/lib)
  endif()
  target_include_directories(libc.utils.MPFRWrapper PUBLIC ${LIBC_SOURCE_DIR})
  target_link_libraries(libc.utils.MPFRWrapper PUBLIC libc.utils.MPCommon)
elseif(NOT LIBC_TARGET_OS_IS_GPU AND NOT LLVM_LIBC_FULL_BUILD)
  message(WARNING "Math tests using MPFR will be skipped.")
endif()
