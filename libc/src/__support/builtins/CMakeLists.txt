add_header_library(
  fpconvert_helper
  HDRS
    fpconvert_helper.h
  DEPENDS
    libc.hdr.fenv_macros
    libc.src.__support.CPP.algorithm
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.type_traits
    libc.src.__support.FPUtil.dyadic_float
    libc.src.__support.FPUtil.fenv_impl
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.macros.attributes
    libc.src.__support.macros.config
)

add_header_library(
  addtf3
  HDRS
    addtf3.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.FPUtil.generic.add_sub
    libc.src.__support.macros.config
)

add_header_library(
  subtf3
  HDRS
    subtf3.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.FPUtil.generic.add_sub
    libc.src.__support.macros.config
)

add_header_library(
  multf3
  HDRS
    multf3.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.FPUtil.generic.mul
    libc.src.__support.macros.config
)

add_header_library(
  divtf3
  HDRS
    divtf3.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.FPUtil.generic.div
    libc.src.__support.macros.config
)

add_header_library(
  adddf3
  HDRS
    adddf3.h
  DEPENDS
    libc.src.__support.FPUtil.generic.add_sub
    libc.src.__support.macros.config
)

add_header_library(
  subdf3
  HDRS
    subdf3.h
  DEPENDS
    libc.src.__support.FPUtil.generic.add_sub
    libc.src.__support.macros.config
)

add_header_library(
  muldf3
  HDRS
    muldf3.h
  DEPENDS
    libc.src.__support.FPUtil.generic.mul
    libc.src.__support.macros.config
)

add_header_library(
  divdf3
  HDRS
    divdf3.h
  DEPENDS
    libc.src.__support.FPUtil.generic.div
    libc.src.__support.macros.config
)

add_header_library(
  addsf3
  HDRS
    addsf3.h
  DEPENDS
    libc.src.__support.FPUtil.generic.add_sub
    libc.src.__support.macros.config
)

add_header_library(
  subsf3
  HDRS
    subsf3.h
  DEPENDS
    libc.src.__support.FPUtil.generic.add_sub
    libc.src.__support.macros.config
)

add_header_library(
  mulsf3
  HDRS
    mulsf3.h
  DEPENDS
    libc.src.__support.FPUtil.generic.mul
    libc.src.__support.macros.config
)

add_header_library(
  divsf3
  HDRS
    divsf3.h
  DEPENDS
    libc.src.__support.FPUtil.generic.div
    libc.src.__support.macros.config
)

add_header_library(
  extendsftf2
  HDRS
    extendsftf2.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.builtins.fpconvert_helper
    libc.src.__support.macros.config
)

add_header_library(
  extenddftf2
  HDRS
    extenddftf2.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.builtins.fpconvert_helper
    libc.src.__support.macros.config
)

add_header_library(
  extendxftf2
  HDRS
    extendxftf2.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.builtins.fpconvert_helper
    libc.src.__support.macros.config
    libc.src.__support.macros.properties.types
)

add_header_library(
  trunctfdf2
  HDRS
    trunctfdf2.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.builtins.fpconvert_helper
    libc.src.__support.macros.config
)

add_header_library(
  trunctfsf2
  HDRS
    trunctfsf2.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.builtins.fpconvert_helper
    libc.src.__support.macros.config
)

add_header_library(
  trunctfxf2
  HDRS
    trunctfxf2.h
  DEPENDS
    libc.include.llvm-libc-types.float128
    libc.src.__support.builtins.fpconvert_helper
    libc.src.__support.macros.config
    libc.src.__support.macros.properties.types
)

add_header_library(
  extendsfdf2
  HDRS
    extendsfdf2.h
  DEPENDS
    libc.src.__support.builtins.fpconvert_helper
    libc.src.__support.macros.config
)

add_header_library(
  truncdfsf2
  HDRS
    truncdfsf2.h
  DEPENDS
    libc.src.__support.builtins.fpconvert_helper
    libc.src.__support.macros.config
)
