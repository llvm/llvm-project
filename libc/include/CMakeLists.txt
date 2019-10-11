
add_header(
  llvm_libc_common_h
  HDR
    __llvm-libc-common.h
)

add_header(
  ctype_h
  HDR
   ctype.h
  DEPENDS
    llvm_libc_common_h
)

add_header(
  math_h
  HDR
    math.h
  DEPENDS
    llvm_libc_common_h
)

add_header(
  string_h
  HDR
    string.h
  DEPENDS
    llvm_libc_common_h
)
