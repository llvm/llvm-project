add_entrypoint_object(
  setjmp
  SRCS
    setjmp.cpp
  HDRS
    ../setjmp_impl.h
  DEPENDS
    libc.hdr.types.jmp_buf
  COMPILE_OPTIONS
    -O3
    -fno-omit-frame-pointer
    # TODO: Remove once one of these lands:
    # https://github.com/llvm/llvm-project/pull/87837
    # https://github.com/llvm/llvm-project/pull/88054
    # https://github.com/llvm/llvm-project/pull/88157
    -ftrivial-auto-var-init=uninitialized
)

add_entrypoint_object(
  longjmp
  SRCS
    longjmp.cpp
  HDRS
    ../longjmp.h
  DEPENDS
    libc.hdr.types.jmp_buf
  COMPILE_OPTIONS
    -O3
    -fomit-frame-pointer
)
