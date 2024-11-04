# Helper function to set up dependencies if they exist.
function(add_macro_header name)
  cmake_parse_arguments(
    "MACRO_HEADER"
    "" # Optional arguments
    "HDR" # Single value arguments
    "DEPENDS" # Multi-value arguments
    ${ARGN}
  )
  if(TARGET libc.include.llvm-libc-macros.${LIBC_TARGET_OS}.${name})
    add_header(
      ${name}
      HDR
        ${MACRO_HEADER_HDR}
      DEPENDS
        .${LIBC_TARGET_OS}.${name}
        ${MACRO_HEADER_DEPENDS}
    )
  else()
    add_header(
      ${name}
      HDR
        ${MACRO_HEADER_HDR}
      DEPENDS
        ${MACRO_HEADER_DEPENDS}
    )
  endif()
endfunction(add_macro_header)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_macro_header(
  generic_error_number_macros
  HDR
    generic-error-number-macros.h
)

add_macro_header(
  null_macro
  HDR
    null-macro.h
)

add_macro_header(
  fcntl_macros
  HDR
    fcntl-macros.h
)

add_macro_header(
  features_macros
  HDR
    features-macros.h
)

add_macro_header(
  fenv_macros
  HDR
    fenv-macros.h
)

add_macro_header(
  file_seek_macros
  HDR
    file-seek-macros.h
)

add_macro_header(
  float_macros
  HDR
    float-macros.h
)

add_macro_header(
  math_macros
  HDR
    math-macros.h
)

add_macro_header(
  offsetof_macro
  HDR
    offsetof-macro.h
)

add_macro_header(
  containerof_macro
  HDR
    containerof-macro.h
  DEPENDS
    .offsetof_macro
)

add_macro_header(
  sched_macros
  HDR
    sched-macros.h
)

add_macro_header(
  signal_macros
  HDR
    signal-macros.h
)

add_macro_header(
  stdio_macros
  HDR
    stdio-macros.h
)

add_macro_header(
  stdlib_macros
  HDR
    stdlib-macros.h
)

add_macro_header(
  sys_auxv_macros
  HDR
    sys-auxv-macros.h
)

add_macro_header(
  sys_ioctl_macros
  HDR
    sys-ioctl-macros.h
)

add_macro_header(
  sys_stat_macros
  HDR
    sys-stat-macros.h
)

add_macro_header(
  sys_mman_macros
  HDR
    sys-mman-macros.h
)

add_macro_header(
  sys_queue_macros
  HDR
    sys-queue-macros.h
  DEPENDS
    .null_macro
    .containerof_macro
)

add_macro_header(
  sys_random_macros
  HDR
    sys-random-macros.h
)

add_macro_header(
  sys_resource_macros
  HDR
    sys-resource-macros.h
)

add_macro_header(
  sys_select_macros
  HDR
    sys-select-macros.h
)

add_macro_header(
  sys_socket_macros
  HDR
    sys-socket-macros.h
)

add_macro_header(
  sys_time_macros
  HDR
    sys-time-macros.h
)

add_macro_header(
  sys_wait_macros
  HDR
    sys-wait-macros.h
)

add_macro_header(
  termios_macros
  HDR
    termios-macros.h
)

add_macro_header(
  time_macros
  HDR
    time-macros.h
)

add_macro_header(
  unistd_macros
  HDR
    unistd-macros.h
)

add_macro_header(
  wchar_macros
  HDR
    wchar-macros.h
)
