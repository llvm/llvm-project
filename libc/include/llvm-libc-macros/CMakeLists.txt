add_subdirectory(linux)

add_header(
  fcntl_macros
  HDR
    fcntl-macros.h
  DEPENDS
    .linux.fcntl_macros
)

add_header(
  stdio_macros
  HDR
    stdio-macros.h
)
