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
  sched_macros
  HDR
    sched-macros.h
  DEPENDS
    .linux.sched_macros
)

add_header(
  signal_macros
  HDR
    signal-macros.h
  DEPENDS
    .linux.signal_macros
)

add_header(
  stdio_macros
  HDR
    stdio-macros.h
)

add_header(
  stdlib_macros
  HDR
    stdlib-macros.h
)

add_header(
  sys_auxv_macros
  HDR
    sys-auxv-macros.h
)

add_header(
  sys_ioctl_macros
  HDR
    sys-ioctl-macros.h
  DEPENDS
    .linux.sys_ioctl_macros
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
  sys_random_macros
  HDR
    sys-random-macros.h
  DEPENDS
    .linux.sys_random_macros
)

add_header(
  sys_resource_macros
  HDR
    sys-resource-macros.h
  DEPENDS
    .linux.sys_resource_macros
)

add_header(
  sys_select_macros
  HDR
    sys-select-macros.h
)

add_header(
  sys_time_macros
  HDR
    sys-time-macros.h
  DEPENDS
    .linux.sys_time_macros
)

add_header(
  sys_wait_macros
  HDR
    sys-wait-macros.h
  DEPENDS
    .linux.sys_wait_macros
)

add_header(
  termios_macros
  HDR
    termios-macros.h
  DEPENDS
    .linux.termios_macros
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
