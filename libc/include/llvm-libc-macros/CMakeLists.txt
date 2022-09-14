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

add_header(
  sys_stat_macros
  HDR
    sys-stat-macros.h
  DEPENDS
    .linux.sys_stat_macros
)

add_header(
  unistd_macros
  HDR
    unistd-macros.h
  DEPENDS
    .linux.unistd_macros
)
