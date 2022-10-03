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
  signal_macros
  HDR
    signal-macros.h
  DEPENDS
    .linux.signal_macros
)

add_header(
  sys_stat_macros
  HDR
    sys-stat-macros.h
  DEPENDS
    .linux.sys_stat_macros
)

add_header(
  sys_mman_macros
  HDR
    sys-mman-macros.h
  DEPENDS
    .linux.sys_mman_macros
)

add_header(
  sys_resource_macros
  HDR
    sys-resource-macros.h
  DEPENDS
    .linux.sys_resource_macros
)

add_header(
  time_macros
  HDR
    time-macros.h
  DEPENDS
    .linux.time_macros
)

add_header(
  unistd_macros
  HDR
    unistd-macros.h
  DEPENDS
    .linux.unistd_macros
)
