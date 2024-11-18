if(LIBC_CONF_SETJMP_FORTIFICATION)
  set (setjmp_src setjmp_fortified.cpp)
  set (longjmp_src longjmp_fortified.cpp)
else()
  set (setjmp_src setjmp.cpp)
  set (longjmp_src longjmp.cpp)
endif()

add_entrypoint_object(
  setjmp
  SRCS
    ${setjmp_src}
  HDRS
    ../setjmp_impl.h
  DEPENDS
    libc.hdr.types.jmp_buf
    ${fortification_deps}
  COMPILE_OPTIONS
    -O3
    ${fortification_defs}
)

add_entrypoint_object(
  longjmp
  SRCS
    ${longjmp_src}
  HDRS
    ../longjmp.h
  DEPENDS
    libc.hdr.types.jmp_buf
    ${fortification_deps}
  COMPILE_OPTIONS
    -O3
    -fomit-frame-pointer
    ${fortification_defs}
)
