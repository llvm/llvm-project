add_header_library(
  floatint_helper
  HDRS
    floatint_helper.h
  DEPENDS
    libc.src.__support.CPP.type_traits
    libc.src.__support.FPUtil.dyadic_float
    libc.src.__support.macros.attributes
    libc.src.__support.macros.config
    libc.src.__support.sign
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
  floatsidf
  HDRS
    floatsidf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
)

add_header_library(
  floatdidf
  HDRS
    floatdidf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
)

add_header_library(
  floattidf
  HDRS
    floattidf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
    libc.src.__support.macros.properties.types
)

add_header_library(
  floatsisf
  HDRS
    floatsisf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
)

add_header_library(
  floatdisf
  HDRS
    floatdisf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
)

add_header_library(
  floattisf
  HDRS
    floattisf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
    libc.src.__support.macros.properties.types
)

add_header_library(
  floatunsidf
  HDRS
    floatunsidf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
)

add_header_library(
  floatundidf
  HDRS
    floatundidf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
)

add_header_library(
  floatuntidf
  HDRS
    floatuntidf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
    libc.src.__support.macros.properties.types
)

add_header_library(
  floatunsisf
  HDRS
    floatunsisf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
)

add_header_library(
  floatundisf
  HDRS
    floatundisf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
)

add_header_library(
  floatuntisf
  HDRS
    floatuntisf.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.builtins.floatint_helper
    libc.src.__support.macros.config
    libc.src.__support.macros.properties.types
)
