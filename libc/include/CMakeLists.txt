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

macro(add_header_macro NAME YAML_FILE DEF_FILE GEN_HDR DEPENDS)
  if (LIBC_USE_NEW_HEADER_GEN)
    add_gen_header2(
      ${NAME}
      ${YAML_FILE}
      ${DEF_FILE}
      ${GEN_HDR}
      ${DEPENDS}
    )
  else()
    add_gen_header(
      ${NAME}
      ${DEF_FILE}
      ${GEN_HDR}
      ${DEPENDS}
    )
  endif()
endmacro()

add_header_macro(
  ctype
  YAML_FILE ../libc/newhdrgen/yaml/ctype.yaml
  DEF_FILE ctype.h.def
  GEN_HDR ctype.h
  DEPENDS
    .llvm_libc_common_h
)

add_header_macro(
  dirent
  YAML_FILE ../libc/newhdrgen/yaml/dirent.yaml
  DEF_FILE dirent.h.def
  GEN_HDR dirent.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.ino_t
    .llvm-libc-types.DIR
    .llvm-libc-types.struct_dirent
)

add_header_macro(
  fcntl
  YAML_FILE ../libc/newhdrgen/yaml/fcntl.yaml
  DEF_FILE fcntl.h.def
  GEN_HDR fcntl.h
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
  YAML_FILE ../libc/newhdrgen/yaml/dlfcn.yaml
  DEF_FILE dlfcn.h.def
  GEN_HDR dlfcn.h
  DEPENDS
    .llvm-libc-macros.dlfcn_macros
    .llvm_libc_common_h
)

add_header_macro(
  features
  YAML_FILE ../libc/newhdrgen/yaml/features.yaml
  DEF_FILE features.h.def
  GEN_HDR features.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.features_macros
)

add_header_macro(
  fenv
  YAML_FILE ../libc/newhdrgen/yaml/fenv.yaml
  DEF_FILE fenv.h.def
  GEN_HDR fenv.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.fenv_macros
    .llvm-libc-types.fenv_t
    .llvm-libc-types.fexcept_t
)

add_header_macro(
  inttypes
  YAML_FILE ../libc/newhdrgen/yaml/inttypes.yaml
  DEF_FILE inttypes.h.def
  GEN_HDR inttypes.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.imaxdiv_t
    .llvm-libc-macros.inttypes_macros
)

add_header_macro(
  float
  YAML_FILE ../libc/newhdrgen/yaml/float.yaml
  DEF_FILE float.h.def
  GEN_HDR float.h
  DEPENDS
    .llvm-libc-macros.float_macros
)

add_header_macro(
  stdint
  YAML_FILE ../libc/newhdrgen/yaml/stdint.yaml
  DEF_FILE stdint.h.def
  GEN_HDR stdint.h
  DEPENDS
    .llvm-libc-macros.stdint_macros
)

add_header_macro(
  limits
  YAML_FILE ../libc/newhdrgen/yaml/limits.yaml
  DEF_FILE limits.h.def
  GEN_HDR limits.h
  DEPENDS
    .llvm-libc-macros.limits_macros
)

add_header_macro(
  math
  YAML_FILE ../libc/newhdrgen/yaml/math.yaml
  DEF_FILE math.h.def
  GEN_HDR math.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.float16_macros
    .llvm-libc-macros.math_macros
    .llvm-libc-types.double_t
    .llvm-libc-types.float_t
    .llvm-libc-types.float128
)

add_header_macro(
  stdfix
  YAML_FILE ../libc/newhdrgen/yaml/stdfix.yaml
  DEF_FILE stdfix.h.def
  GEN_HDR stdfix.h
  DEPENDS
    .llvm-libc-macros.stdfix_macros
)

# TODO: This should be conditional on POSIX networking being included.
file(MAKE_DIRECTORY ${LIBC_INCLUDE_DIR}/arpa)

add_header_macro(
  arpa_inet
  YAML_FILE ../libc/newhdrgen/yaml/arpa_inet.yaml
  DEF_FILE arpa/inet.h.def
  GEN_HDR arpa/inet.h
  DEPENDS
    .llvm_libc_common_h
)

add_header_macro(
  assert
  YAML_FILE ../libc/newhdrgen/yaml/assert.yaml
  DEF_FILE assert.h.def
  GEN_HDR assert.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.assert_macros
)

add_header_macro(
  setjmp
  YAML_FILE ../libc/newhdrgen/yaml/setjmp.yaml
  DEF_FILE setjmp.h.def
  GEN_HDR setjmp.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.jmp_buf
)

add_header_macro(
  string
  YAML_FILE ../libc/newhdrgen/yaml/string.yaml
  DEF_FILE string.h.def
  GEN_HDR string.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.null_macro
    .llvm-libc-types.size_t
)

add_header_macro(
  strings
  YAML_FILE ../libc/newhdrgen/yaml/strings.yaml
  DEF_FILE strings.h.def
  GEN_HDR strings.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.size_t
)

add_header_macro(
  search
  YAML_FILE ../libc/newhdrgen/yaml/search.yaml
  DEF_FILE search.h.def
  GEN_HDR search.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.ACTION
    .llvm-libc-types.ENTRY
    .llvm-libc-types.struct_hsearch_data
    .llvm-libc-types.size_t
)

add_header_macro(
  time
  YAML_FILE ../libc/newhdrgen/yaml/time.yaml
  DEF_FILE time.h.def
  GEN_HDR time.h
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
  YAML_FILE ../libc/newhdrgen/yaml/threads.yaml
  DEF_FILE threads.h.def
  GEN_HDR threads.h
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
  YAML_FILE ../libc/newhdrgen/yaml/errno.yaml
  DEF_FILE errno.h.def
  GEN_HDR errno.h
  DEPENDS
    .llvm-libc-macros.generic_error_number_macros
    .llvm-libc-macros.error_number_macros
)

add_header_macro(
  signal
  YAML_FILE ../libc/newhdrgen/yaml/signal.yaml
  DEF_FILE signal.h.def
  GEN_HDR signal.h
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
  YAML_FILE ../libc/newhdrgen/yaml/stdbit.yaml
  DEF_FILE stdbit.h.def
  GEN_HDR stdbit.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.stdbit_macros
)

add_header_macro(
  stdckdint
  YAML_FILE ../libc/newhdrgen/yaml/stdckdint.yaml
  DEF_FILE stdckdint.h.def
  GEN_HDR stdckdint.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.stdckdint_macros
)

add_header_macro(
  stdio
  YAML_FILE ../libc/newhdrgen/yaml/stdio.yaml
  DEF_FILE stdio.h.def
  GEN_HDR stdio.h
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
  YAML_FILE ../libc/newhdrgen/yaml/stdlib.yaml
  DEF_FILE stdlib.h.def
  GEN_HDR stdlib.h
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
  YAML_FILE ../libc/newhdrgen/yaml/unistd.yaml
  DEF_FILE unistd.h.def
  GEN_HDR unistd.h
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
  YAML_FILE ../libc/newhdrgen/yaml/pthread.yaml
  DEF_FILE pthread.h.def
  GEN_HDR pthread.h
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
    .llvm-libc-types.pthread_t
)

add_header_macro(
  sched
  YAML_FILE ../libc/newhdrgen/yaml/sched.yaml
  DEF_FILE sched.h.def
  GEN_HDR sched.h
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
  YAML_FILE ../libc/newhdrgen/yaml/spawn.yaml
  DEF_FILE spawn.h.def
  GEN_HDR spawn.h
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
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_auxv.yaml
  DEF_FILE sys/auxv.h.def
  GEN_HDR sys/auxv.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_auxv_macros
)

add_header_macro(
  sys_epoll
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_epoll.yaml
  DEF_FILE sys/epoll.h.def
  GEN_HDR sys/epoll.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.struct_epoll_event
    .llvm-libc-types.struct_epoll_data
    .llvm-libc-types.sigset_t
    .llvm-libc-macros.sys_epoll_macros
)

add_header_macro(
  sys_ioctl
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_ioctl.yaml
  DEF_FILE sys/ioctl.h.def
  GEN_HDR sys/ioctl.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_ioctl_macros
)

add_header_macro(
  sys_mman
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_mman.yaml
  DEF_FILE sys/mman.h.def
  GEN_HDR sys/mman.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_mman_macros
    .llvm-libc-types.off_t
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
)

add_header_macro(
  sys_prctl
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_prctl.yaml
  DEF_FILE sys/prctl.h.def
  GEN_HDR sys/prctl.h
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
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_random.yaml
  DEF_FILE sys/random.h.def
  GEN_HDR sys/random.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_random_macros
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
)

add_header_macro(
  sys_resource
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_resource.yaml
  DEF_FILE sys/resource.h.def
  GEN_HDR sys/resource.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_resource_macros
    .llvm-libc-types.rlim_t
    .llvm-libc-types.struct_rlimit
)

add_header_macro(
  sys_stat
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_stat.yaml
  DEF_FILE sys/stat.h.def
  GEN_HDR sys/stat.h
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
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_select.yaml
  DEF_FILE sys/select.h.def
  GEN_HDR sys/select.h
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
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_sendfile.yaml
  DEF_FILE sys/sendfile.h.def
  GEN_HDR sys/sendfile.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.off_t
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
)

add_header_macro(
  sys_socket
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_socket.yaml
  DEF_FILE sys/socket.h.def
  GEN_HDR sys/socket.h
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
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_statvfs.yaml
  DEF_FILE sys/statvfs.h.def
  GEN_HDR sys/statvfs.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.struct_statvfs
)

add_header_macro(
  sys_syscall
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_syscall.yaml
  DEF_FILE sys/syscall.h.def
  GEN_HDR sys/syscall.h
)

add_header_macro(
  sys_time
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_time.yaml
  DEF_FILE sys/time.h.def
  GEN_HDR sys/time.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.struct_timeval
    .llvm-libc-macros.sys_time_macros
)

add_header_macro(
  sys_types
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_types.yaml
  DEF_FILE sys/types.h.def
  GEN_HDR sys/types.h
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
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_utsname.yaml
  DEF_FILE sys/utsname.h.def
  GEN_HDR sys/utsname.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.struct_utsname
)

add_header_macro(
  sys_wait
  YAML_FILE ../libc/newhdrgen/yaml/sys/sys_wait.yaml
  DEF_FILE sys/wait.h.def
  GEN_HDR sys/wait.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.sys_wait_macros
    .llvm-libc-types.pid_t
    .llvm-libc-types.struct_rusage
    .llvm-libc-types.siginfo_t
)

add_header_macro(
  termios
  YAML_FILE ../libc/newhdrgen/yaml/termios.yaml
  DEF_FILE termios.h.def
  GEN_HDR termios.h
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
  YAML_FILE ../libc/newhdrgen/yaml/uchar.yaml
  DEF_FILE uchar.h.def
  GEN_HDR uchar.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.mbstate_t
    .llvm-libc-types.char8_t
    .llvm-libc-types.char16_t
    .llvm-libc-types.char32_t
)

add_header_macro(
  wchar
  YAML_FILE ../libc/newhdrgen/yaml/wchar.yaml
  DEF_FILE wchar.h.def
  GEN_HDR wchar.h
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
    YAML_FILE ../libc/newhdrgen/yaml/rpc.yaml
    DEF_FILE gpu/rpc.h.def
    GEN_HDR gpu/rpc.h
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
