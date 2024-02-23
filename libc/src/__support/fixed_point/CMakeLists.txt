add_header_library(
  fx_rep
  HDRS
    fx_rep.h
  DEPENDS
    libc.include.llvm-libc-macros.stdfix_macros
    libc.src.__support.macros.attributes
)
