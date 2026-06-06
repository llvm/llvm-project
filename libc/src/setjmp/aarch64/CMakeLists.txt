if(LIBC_CONF_SETJMP_AARCH64_RESTORE_PLATFORM_REGISTER)
  libc_set_definition(setjmp_config_options "LIBC_COPT_SETJMP_AARCH64_RESTORE_PLATFORM_REGISTER")
endif()

add_entrypoint_object(
  setjmp
  SRCS
    setjmp.cpp
  HDRS
    ../setjmp_impl.h
  COMPILE_OPTIONS
    "${setjmp_config_options}"
  DEPENDS
    libc.hdr.types.jmp_buf
)

add_entrypoint_object(
  longjmp
  SRCS
    longjmp.cpp
  HDRS
    ../longjmp.h
  COMPILE_OPTIONS
    "${setjmp_config_options}"
  DEPENDS
    libc.hdr.types.jmp_buf
)

add_entrypoint_object(
  sigsetjmp
  SRCS
    sigsetjmp.cpp
  HDRS
    ../sigsetjmp.h
  DEPENDS
    libc.hdr.types.sigjmp_buf
    libc.hdr.types.sigset_t
    libc.hdr.offsetof_macros
    libc.src.setjmp.sigsetjmp_epilogue
    libc.src.setjmp.setjmp
)
