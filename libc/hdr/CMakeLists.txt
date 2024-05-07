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

add_proxy_header_library(
  fcntl_macros
  HDRS
    fcntl_macros.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-macros.fcntl_macros
    libc.include.fcntl
)

add_proxy_header_library(
  fenv_macros
  HDRS
    fenv_macros.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-macros.fenv_macros
    libc.include.fenv
)

add_proxy_header_library(
  signal_macros
  HDRS
    signal_macros.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-macros.signal_macros
    libc.include.signal
)

add_proxy_header_library(
  sys_epoll_macros
  HDRS
    sys_epoll_macros.h
  FULL_BUILD_DEPENDS
    libc.include.sys_epoll
    libc.include.llvm-libc-macros.sys_epoll_macros
)

add_subdirectory(types)
