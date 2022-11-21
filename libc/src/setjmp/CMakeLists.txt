add_entrypoint_object(
  setjmp
  SRCS
    setjmp.cpp
  HDRS
    setjmp_impl.h
  COMPILE_OPTIONS
    -O3 # We do not want any local variables in setjmp
    -fno-omit-frame-pointer # The implementation assumes frame pointer on to the stack
  DEPENDS
    libc.include.setjmp
)

add_entrypoint_object(
  longjmp
  SRCS
    longjmp.cpp
  HDRS
    longjmp.h
  COMPILE_OPTIONS
    -O3 # We do not want any local variables in longjmp
  DEPENDS
    libc.include.setjmp
)
