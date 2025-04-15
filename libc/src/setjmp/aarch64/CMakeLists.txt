if(LIBC_CONF_SETJMP_AARCH64_RESTORE_PLATFORM_REGISTER)
  list(APPEND setjmp_config_options "-DLIBC_COPT_SETJMP_AARCH64_RESTORE_PLATFORM_REGISTER")
endif()
if(setjmp_config_options)
  list(PREPEND setjmp_config_options "COMPILE_OPTIONS")
endif()

add_entrypoint_object(
  setjmp
  SRCS
    setjmp.cpp
  HDRS
    ../setjmp_impl.h
  DEPENDS
    libc.hdr.types.jmp_buf
  ${setjmp_config_options}
)

add_entrypoint_object(
  longjmp
  SRCS
    longjmp.cpp
  HDRS
    ../longjmp.h
  DEPENDS
    libc.hdr.types.jmp_buf
  ${setjmp_config_options}
)
