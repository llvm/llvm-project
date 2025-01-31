if(LIBC_TESTS_CAN_USE_MPFR)
  set(LIBC_MPC_INSTALL_PATH "" CACHE PATH "Path to where MPC is installed (e.g. C:/src/install or ~/src/install)")

  if(LIBC_MPC_INSTALL_PATH)
    set(LIBC_TESTS_CAN_USE_MPC TRUE)
  elseif(LIBC_TARGET_OS_IS_GPU OR LLVM_LIBC_FULL_BUILD)
    # In full build mode, the MPC library should be built using our own facilities,
    # which is currently not possible.
    set(LIBC_TESTS_CAN_USE_MPC FALSE)
  else()
    try_compile(
      LIBC_TESTS_CAN_USE_MPC
      ${CMAKE_CURRENT_BINARY_DIR}
      SOURCES
        ${LIBC_SOURCE_DIR}/utils/MPCWrapper/check_mpc.cpp
      COMPILE_DEFINITIONS
        ${LIBC_COMPILE_OPTIONS_DEFAULT}
      LINK_LIBRARIES
        -lmpc -lmpfr -lgmp -latomic
    )
  endif()
endif()
