function(add_proxy_header_library target_name)
  cmake_parse_arguments(
    "ADD_PROXY_HEADER"
    "" # Optional arguments
    "" # Single value arguments
    "DEPENDS;FULL_BUILD_DEPENDS" # Multi-value arguments
    ${ARGN}
  )

  set(deps "")
  if(ADD_PROXY_HEADER_DEPENDS)
    list(APPEND deps ${ADD_PROXY_HEADER_DEPENDS})
  endif()

  if(LLVM_LIBC_FULL_BUILD AND ADD_PROXY_HEADER_FULL_BUILD_DEPENDS)
    list(APPEND deps ${ADD_PROXY_HEADER_FULL_BUILD_DEPENDS})
  endif()

  add_header_library(
    ${target_name}
    ${ADD_PROXY_HEADER_UNPARSED_ARGUMENTS}
    DEPENDS ${deps}
  )
endfunction()

add_proxy_header_library(
  math_macros
  HDRS
    math_macros.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-macros.math_macros
    libc.include.math
)
