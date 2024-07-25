set(LIBC_INCLUDE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBC_INCLUDE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(LLVMLibCHeaderRules)

# The GPU build wants to install files in the compiler's resource directory.
if(LIBC_TARGET_OS_IS_GPU)
  include(GetClangResourceDir)
endif()

add_subdirectory(llvm-libc-macros)
add_subdirectory(llvm-libc-types)

add_header(
  llvm_libc_common_h
  HDR
    __llvm-libc-common.h
)

macro(add_header_macro TARGET_NAME YAML_FILE DEF_FILE GEN_HDR DEPENDS)
  if (LIBC_USE_NEW_HEADER_GEN)
    add_gen_header2(
      ${TARGET_NAME}
      YAML_FILE ${YAML_FILE}
      DEF_FILE ${DEF_FILE}
      GEN_HDR ${GEN_HDR}
      ${DEPENDS}
      ${ARGN}
    )
  else()
    add_gen_header(
      ${TARGET_NAME}
      DEF_FILE ${DEF_FILE}
      GEN_HDR ${GEN_HDR}
      ${DEPENDS}
      ${ARGN}
    )
  endif()
endmacro()

add_header_macro(
  ctype
  ../libc/newhdrgen/yaml/ctype.yaml
  ctype.h.def
  ctype.h
  DEPENDS
    .llvm_libc_common_h
)

add_header_macro(
  dirent
  ../libc/newhdrgen/yaml/dirent.yaml
  dirent.h.def
  dirent.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.ino_t
    .llvm-libc-types.DIR
    .llvm-libc-types.struct_dirent
)

add_header_macro(
  fcntl
  ../libc/newhdrgen/yaml/fcntl.yaml
  fcntl.h.def
  fcntl.h
  DEPENDS
    .llvm-libc-macros.fcntl_macros
    .llvm-libc-types.mode_t
    .llvm-libc-types.struct_flock
    .llvm-libc-types.struct_flock64
    .llvm-libc-types.off64_t
    .llvm-libc-types.pid_t
    .llvm-libc-types.off_t
    .llvm_libc_common_h
)

add_header_macro(
  dlfcn
  ../libc/newhdrgen/yaml/dlfcn.yaml
  dlfcn.h.def
  dlfcn.h
  DEPENDS
    .llvm-libc-macros.dlfcn_macros
    .llvm_libc_common_h
)

add_header_macro(
  features
  ../libc/newhdrgen/yaml/features.yaml
  features.h.def
  features.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.features_macros
)

add_header_macro(
  fenv
  ../libc/newhdrgen/yaml/fenv.yaml
  fenv.h.def
  fenv.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.fenv_macros
    .llvm-libc-types.fenv_t
    .llvm-libc-types.fexcept_t
)

add_header_macro(
  inttypes
  ../libc/newhdrgen/yaml/inttypes.yaml
  inttypes.h.def
  inttypes.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.imaxdiv_t
    .llvm-libc-macros.inttypes_macros
)

add_header_macro(
  float
  ../libc/newhdrgen/yaml/float.yaml
  float.h.def
  float.h
  DEPENDS
    .llvm-libc-macros.float_macros
)

add_header_macro(
  stdint
  ../libc/newhdrgen/yaml/stdint.yaml
  stdint.h.def
  stdint.h
  DEPENDS
    .llvm-libc-macros.stdint_macros
)

add_header_macro(
  limits
  ../libc/newhdrgen/yaml/limits.yaml
  limits.h.def
  limits.h
  DEPENDS
    .llvm-libc-macros.limits_macros
)

add_header_macro(
  math
  ../libc/newhdrgen/yaml/math.yaml
  math.h.def
  math.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.float16_macros
    .llvm-libc-macros.math_macros
    .llvm-libc-macros.math_function_macros
    .llvm-libc-types.double_t
    .llvm-libc-types.float_t
    .llvm-libc-types.float128
)

add_header_macro(
  stdfix
  ../libc/newhdrgen/yaml/stdfix.yaml
  stdfix.h.def
  stdfix.h
  DEPENDS
    .llvm-libc-macros.stdfix_macros
)

# TODO: This should be conditional on POSIX networking being included.
file(MAKE_DIRECTORY ${LIBC_INCLUDE_DIR}/arpa)

add_header_macro(
  arpa_inet
  ../libc/newhdrgen/yaml/arpa/inet.yaml
  arpa/inet.h.def
  arpa/inet.h
  DEPENDS
    .llvm_libc_common_h
)

add_header_macro(
  assert
  ../libc/newhdrgen/yaml/assert.yaml
  assert.h.def
  assert.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.assert_macros
)

add_header_macro(
  setjmp
  ../libc/newhdrgen/yaml/setjmp.yaml
  setjmp.h.def
  setjmp.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.jmp_buf
)

add_header_macro(
  string
  ../libc/newhdrgen/yaml/string.yaml
  string.h.def
  string.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.null_macro
    .llvm-libc-types.size_t
)

add_header_macro(
  strings
  ../libc/newhdrgen/yaml/strings.yaml
  strings.h.def
  strings.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.size_t
)

add_header_macro(
  search
  ../libc/newhdrgen/yaml/search.yaml
  search.h.def
  search.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.ACTION
    .llvm-libc-types.ENTRY
    .llvm-libc-types.struct_hsearch_data
    .llvm-libc-types.size_t
)

add_header_macro(
  time
  ../libc/newhdrgen/yaml/time.yaml
  time.h.def
  time.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.time_macros
    .llvm-libc-types.clock_t
    .llvm-libc-types.time_t
    .llvm-libc-types.struct_tm
    .llvm-libc-types.struct_timespec
    .llvm-libc-types.struct_timeval
    .llvm-libc-types.clockid_t
)

add_header_macro(
  threads
  ../libc/newhdrgen/yaml/threads.yaml
  threads.h.def
  threads.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.__call_once_func_t
    .llvm-libc-types.once_flag
    .llvm-libc-types.cnd_t
    .llvm-libc-types.mtx_t
    .llvm-libc-types.thrd_t
    .llvm-libc-types.thrd_start_t
    .llvm-libc-types.tss_t
    .llvm-libc-types.tss_dtor_t
)

add_header_macro(
  errno
  ../libc/newhdrgen/yaml/errno.yaml
  errno.h.def
  errno.h
  DEPENDS
    .llvm-libc-macros.generic_error_number_macros
    .llvm-libc-macros.error_number_macros
)

add_header_macro(
  signal
  ../libc/newhdrgen/yaml/signal.yaml
  signal.h.def
  signal.h
  DEPENDS
    .llvm-libc-macros.signal_macros
    .llvm-libc-types.sig_atomic_t
    .llvm-libc-types.sigset_t
    .llvm-libc-types.struct_sigaction
    .llvm-libc-types.union_sigval
    .llvm-libc-types.siginfo_t
    .llvm-libc-types.stack_t
    .llvm-libc-types.pid_t
)

add_header_macro(
  stdbit
  ../libc/newhdrgen/yaml/stdbit.yaml
  stdbit.h.def
  stdbit.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.stdbit_macros
)

add_header_macro(
  stdckdint
  ../libc/newhdrgen/yaml/stdckdint.yaml
  stdckdint.h.def
  stdckdint.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.stdckdint_macros
)

add_header_macro(
  stdio
  ../libc/newhdrgen/yaml/stdio.yaml
  stdio.h.def
  stdio.h
  DEPENDS
    .llvm-libc-macros.file_seek_macros
    .llvm-libc-macros.stdio_macros
    .llvm-libc-types.FILE
    .llvm-libc-types.cookie_io_functions_t
    .llvm-libc-types.off_t
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
    .llvm_libc_common_h
)

add_header_macro(
  stdlib
  ../libc/newhdrgen/yaml/stdlib.yaml
  stdlib.h.def
  stdlib.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.stdlib_macros
    .llvm-libc-types.div_t
    .llvm-libc-types.ldiv_t
    .llvm-libc-types.lldiv_t
    .llvm-libc-types.size_t
    .llvm-libc-types.__bsearchcompare_t
    .llvm-libc-types.__qsortcompare_t
    .llvm-libc-types.__qsortrcompare_t
    .llvm-libc-types.__atexithandler_t
)

add_header_macro(
  unistd
  ../libc/newhdrgen/yaml/unistd.yaml
  unistd.h.def
  unistd.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.file_seek_macros
    .llvm-libc-macros.unistd_macros
    .llvm-libc-types.__exec_argv_t
    .llvm-libc-types.__exec_envp_t
    .llvm-libc-types.off_t
    .llvm-libc-types.pid_t
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
    .llvm-libc-types.uid_t
    .llvm-libc-types.__getoptargv_t
)

add_header_macro(
  pthread
  ../libc/newhdrgen/yaml/pthread.yaml
  pthread.h.def
  pthread.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.__atfork_callback_t
    .llvm-libc-types.__pthread_once_func_t
    .llvm-libc-types.__pthread_start_t
    .llvm-libc-types.__pthread_tss_dtor_t
    .llvm-libc-types.pthread_attr_t
    .llvm-libc-types.pthread_condattr_t
    .llvm-libc-types.pthread_key_t
    .llvm-libc-types.pthread_mutex_t
    .llvm-libc-types.pthread_mutexattr_t
    .llvm-libc-types.pthread_once_t
    .llvm-libc-types.pthread_rwlock_t
    .llvm-libc-types.pthread_rwlockattr_t
    .llvm-libc-types.pthread_spinlock_t
    .llvm-libc-types.pthread_t
)

add_header_macro(
  sched
  ../libc/newhdrgen/yaml/sched.yaml
  sched.h.def
  sched.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sched_macros
    .llvm-libc-types.cpu_set_t
    .llvm-libc-types.pid_t
    .llvm-libc-types.size_t
    .llvm-libc-types.struct_sched_param
    # Needed according to posix standard
    .llvm-libc-types.time_t
    .llvm-libc-types.struct_timespec
)

add_header_macro(
  spawn
  ../libc/newhdrgen/yaml/spawn.yaml
  spawn.h.def
  spawn.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.mode_t
    .llvm-libc-types.pid_t
    .llvm-libc-types.posix_spawnattr_t
    .llvm-libc-types.posix_spawn_file_actions_t
)

# TODO: Not all platforms will have a include/sys directory. Add the sys
# directory and the targets for sys/*.h files conditional to the OS requiring
# them.
file(MAKE_DIRECTORY ${LIBC_INCLUDE_DIR}/sys)

add_header_macro(
  sys_auxv
  ../libc/newhdrgen/yaml/sys/auxv.yaml
  sys/auxv.h.def
  sys/auxv.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_auxv_macros
)

add_header_macro(
  sys_epoll
  ../libc/newhdrgen/yaml/sys/epoll.yaml
  sys/epoll.h.def
  sys/epoll.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.struct_epoll_event
    .llvm-libc-types.struct_epoll_data
    .llvm-libc-types.sigset_t
    .llvm-libc-macros.sys_epoll_macros
)

add_header_macro(
  sys_ioctl
  ../libc/newhdrgen/yaml/sys/ioctl.yaml
  sys/ioctl.h.def
  sys/ioctl.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_ioctl_macros
)

add_header_macro(
  sys_mman
  ../libc/newhdrgen/yaml/sys/mman.yaml
  sys/mman.h.def
  sys/mman.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_mman_macros
    .llvm-libc-types.off_t
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
)

add_header_macro(
  sys_prctl
  ../libc/newhdrgen/yaml/sys/prctl.yaml
  sys/prctl.h.def
  sys/prctl.h
  DEPENDS
    .llvm_libc_common_h
)

add_header(
  sys_queue
  HDR
    sys/queue.h
  DEPENDS
    .llvm-libc-macros.sys_queue_macros
)

add_header_macro(
  sys_random
  ../libc/newhdrgen/yaml/sys/random.yaml
  sys/random.h.def
  sys/random.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_random_macros
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
)

add_header_macro(
  sys_resource
  ../libc/newhdrgen/yaml/sys/resource.yaml
  sys/resource.h.def
  sys/resource.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_resource_macros
    .llvm-libc-types.rlim_t
    .llvm-libc-types.struct_rlimit
)

add_header_macro(
  sys_stat
  ../libc/newhdrgen/yaml/sys/stat.yaml
  sys/stat.h.def
  sys/stat.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_stat_macros
    .llvm-libc-types.mode_t
    .llvm-libc-types.dev_t
    .llvm-libc-types.ino_t
    .llvm-libc-types.nlink_t
    .llvm-libc-types.uid_t
    .llvm-libc-types.gid_t
    .llvm-libc-types.off_t
    .llvm-libc-types.struct_timespec
    .llvm-libc-types.struct_timeval
    .llvm-libc-types.blksize_t
    .llvm-libc-types.blkcnt_t
    .llvm-libc-types.struct_stat
)

add_header_macro(
  sys_select
  ../libc/newhdrgen/yaml/sys/select.yaml
  sys/select.h.def
  sys/select.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_select_macros
    .llvm-libc-types.fd_set
    .llvm-libc-types.sigset_t
    .llvm-libc-types.suseconds_t
    .llvm-libc-types.time_t
    .llvm-libc-types.struct_timespec
    .llvm-libc-types.struct_timeval
)

add_header_macro(
  sys_sendfile
  ../libc/newhdrgen/yaml/sys/sendfile.yaml
  sys/sendfile.h.def
  sys/sendfile.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.off_t
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
)

add_header_macro(
  sys_socket
  ../libc/newhdrgen/yaml/sys/socket.yaml
  sys/socket.h.def
  sys/socket.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_socket_macros
    .llvm-libc-types.sa_family_t
    .llvm-libc-types.socklen_t
    .llvm-libc-types.struct_sockaddr
    .llvm-libc-types.struct_sockaddr_un
)

add_header_macro(
  sys_statvfs
  ../libc/newhdrgen/yaml/sys/statvfs.yaml
  sys/statvfs.h.def
  sys/statvfs.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.struct_statvfs
)

add_header_macro(
  sys_syscall
  ../libc/newhdrgen/yaml/sys/syscall.yaml
  sys/syscall.h.def
  sys/syscall.h
  DEPENDS
)

add_header_macro(
  sys_time
  ../libc/newhdrgen/yaml/sys/time.yaml
  sys/time.h.def
  sys/time.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.struct_timeval
    .llvm-libc-macros.sys_time_macros
)

add_header_macro(
  sys_types
  ../libc/newhdrgen/yaml/sys/types.yaml
  sys/types.h.def
  sys/types.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.blkcnt_t
    .llvm-libc-types.blksize_t
    .llvm-libc-types.clockid_t
    .llvm-libc-types.dev_t
    .llvm-libc-types.gid_t
    .llvm-libc-types.ino_t
    .llvm-libc-types.mode_t
    .llvm-libc-types.nlink_t
    .llvm-libc-types.off_t
    .llvm-libc-types.pid_t
    .llvm-libc-types.pthread_attr_t
    .llvm-libc-types.pthread_key_t
    .llvm-libc-types.pthread_mutex_t
    .llvm-libc-types.pthread_mutexattr_t
    .llvm-libc-types.pthread_once_t
    .llvm-libc-types.pthread_t
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
    .llvm-libc-types.suseconds_t
    .llvm-libc-types.time_t
    .llvm-libc-types.uid_t
)

add_header_macro(
  sys_utsname
  ../libc/newhdrgen/yaml/sys/utsname.yaml
  sys/utsname.h.def
  sys/utsname.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.struct_utsname
)

add_header_macro(
  sys_wait
  ../libc/newhdrgen/yaml/sys/wait.yaml
  sys/wait.h.def
  sys/wait.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_wait_macros
    .llvm-libc-types.pid_t
    .llvm-libc-types.struct_rusage
    .llvm-libc-types.siginfo_t
)

add_header_macro(
  termios
  ../libc/newhdrgen/yaml/termios.yaml
  termios.h.def
  termios.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.termios_macros
    .llvm-libc-types.cc_t
    .llvm-libc-types.pid_t
    .llvm-libc-types.speed_t
    .llvm-libc-types.struct_termios
    .llvm-libc-types.tcflag_t
)

add_header_macro(
  uchar
  ../libc/newhdrgen/yaml/uchar.yaml
  uchar.h.def
  uchar.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.mbstate_t
    .llvm-libc-types.char8_t
    .llvm-libc-types.char16_t
    .llvm-libc-types.char32_t
)

add_header_macro(
  wchar
  ../libc/newhdrgen/yaml/wchar.yaml
  wchar.h.def
  wchar.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.wchar_macros
    .llvm-libc-types.mbstate_t
    .llvm-libc-types.size_t
    .llvm-libc-types.wint_t
    .llvm-libc-types.wchar_t
)

if(LIBC_TARGET_OS_IS_GPU)
  file(MAKE_DIRECTORY ${LIBC_INCLUDE_DIR}/gpu)

  add_header_macro(
    gpu_rpc
    ../libc/newhdrgen/yaml/gpu/rpc.yaml
    gpu/rpc.h.def
    gpu/rpc.h
    DEPENDS
      .llvm_libc_common_h
      .llvm-libc-types.rpc_opcodes_t
  )
endif()

if(NOT LLVM_LIBC_FULL_BUILD)
  # We don't install headers in non-fullbuild mode.
  return()
endif()

function(get_all_install_header_targets out_var)
  set(all_deps ${ARGN})
  foreach(target IN LISTS ARGN)
    get_target_property(deps ${target} DEPS)
    if(NOT deps)
      continue()
    endif()
    list(APPEND all_deps ${deps})
    get_all_install_header_targets(nested_deps ${deps})
    list(APPEND all_deps ${nested_deps})
  endforeach()
  list(REMOVE_DUPLICATES all_deps)
  set(${out_var} ${all_deps} PARENT_SCOPE)
endfunction(get_all_install_header_targets)

get_all_install_header_targets(all_install_header_targets ${TARGET_PUBLIC_HEADERS})
add_library(libc-headers INTERFACE)
add_dependencies(libc-headers ${all_install_header_targets})
target_include_directories(libc-headers SYSTEM INTERFACE ${LIBC_INCLUDE_DIR})

foreach(target IN LISTS all_install_header_targets)
  get_target_property(header_file ${target} HEADER_FILE_PATH)
  if(NOT header_file)
    message(FATAL_ERROR "Installable header file '${target}' does not have the "
                        "HEADER_FILE_PATH property set.")
  endif()
  file(RELATIVE_PATH relative_path ${LIBC_INCLUDE_DIR} ${header_file})
  get_filename_component(nested_dir ${relative_path} DIRECTORY)
  install(FILES ${header_file}
          DESTINATION ${LIBC_INSTALL_INCLUDE_DIR}/${nested_dir}
          COMPONENT libc-headers)
  # The GPU optionally provides the supported declarations externally so
  # offloading languages like CUDA and OpenMP know what is supported by libc. We
  # install these in the compiler's resource directory at a preset location.
  if(LIBC_TARGET_OS_IS_GPU AND PACKAGE_VERSION)
    get_target_property(decls_file ${target} DECLS_FILE_PATH)
    if(NOT decls_file)
      continue()
    endif()
    get_clang_resource_dir(resource_dir SUBDIR include)
    file(RELATIVE_PATH relative_path ${LIBC_INCLUDE_DIR} ${decls_file})
    get_filename_component(nested_dir ${relative_path} DIRECTORY)
    set(install_dir
        ${CMAKE_INSTALL_PREFIX}/${resource_dir}/llvm_libc_wrappers/${nested_dir})
    install(FILES ${decls_file}
            DESTINATION ${install_dir}
            COMPONENT libc-headers)
  endif()
endforeach()

if(LLVM_LIBC_FULL_BUILD)
  add_custom_target(install-libc-headers
                    DEPENDS libc-headers
                    COMMAND "${CMAKE_COMMAND}"
                            -DCMAKE_INSTALL_COMPONENT=libc-headers
                            -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
  # Stripping is a no-op for headers
  add_custom_target(install-libc-headers-stripped DEPENDS install-libc-headers)
endif()
