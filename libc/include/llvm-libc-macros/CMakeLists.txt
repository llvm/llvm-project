add_subdirectory(linux)

add_header(
  fcntl_macros
  HDR
    fcntl-macros.h
  DEPENDS
    .linux.fcntl_macros
)

add_header(
  file_seek_macros
  HDR
    file-seek-macros.h
)
