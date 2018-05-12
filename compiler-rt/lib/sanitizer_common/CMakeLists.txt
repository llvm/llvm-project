# Build system for the common Sanitizer runtime support library components.
# These components are shared between AddressSanitizer and ThreadSanitizer.

set(SANITIZER_SOURCES_NOTERMINATION
  sanitizer_allocator.cc
  sanitizer_common.cc
  sanitizer_deadlock_detector1.cc
  sanitizer_deadlock_detector2.cc
  sanitizer_errno.cc
  sanitizer_file.cc
  sanitizer_flags.cc
  sanitizer_flag_parser.cc
  sanitizer_fuchsia.cc
  sanitizer_libc.cc
  sanitizer_libignore.cc
  sanitizer_linux.cc
  sanitizer_linux_s390.cc
  sanitizer_mac.cc
  sanitizer_openbsd.cc
  sanitizer_persistent_allocator.cc
  sanitizer_platform_limits_linux.cc
  sanitizer_platform_limits_netbsd.cc
  sanitizer_platform_limits_openbsd.cc
  sanitizer_platform_limits_posix.cc
  sanitizer_platform_limits_solaris.cc
  sanitizer_posix.cc
  sanitizer_printf.cc
  sanitizer_procmaps_common.cc
  sanitizer_procmaps_bsd.cc
  sanitizer_procmaps_linux.cc
  sanitizer_procmaps_mac.cc
  sanitizer_procmaps_solaris.cc
  sanitizer_solaris.cc
  sanitizer_stoptheworld_mac.cc
  sanitizer_suppressions.cc
  sanitizer_tls_get_addr.cc
  sanitizer_thread_registry.cc
  sanitizer_win.cc)

if(UNIX AND NOT APPLE AND NOT OS_NAME MATCHES "SunOS")
  list(APPEND SANITIZER_SOURCES_NOTERMINATION
    sanitizer_linux_x86_64.S)
  list(APPEND SANITIZER_SOURCES_NOTERMINATION
    sanitizer_linux_mips64.S)
endif()

set(SANITIZER_SOURCES
  ${SANITIZER_SOURCES_NOTERMINATION} sanitizer_termination.cc)

# Libc functions stubs. These sources should be linked instead of
# SANITIZER_LIBCDEP_SOURCES when sanitizer_common library must not depend on
# libc.
set(SANITIZER_NOLIBC_SOURCES
  sanitizer_common_nolibc.cc)

set(SANITIZER_LIBCDEP_SOURCES
  sanitizer_common_libcdep.cc
  sanitizer_allocator_checks.cc
  sanitizer_linux_libcdep.cc
  sanitizer_mac_libcdep.cc
  sanitizer_posix_libcdep.cc
  sanitizer_stoptheworld_linux_libcdep.cc)

set(SANITIZER_COVERAGE_SOURCES
  sancov_flags.cc
  sanitizer_coverage_fuchsia.cc
  sanitizer_coverage_libcdep_new.cc
  sanitizer_coverage_win_sections.cc)

set(SANITIZER_SYMBOLIZER_SOURCES
  sanitizer_stackdepot.cc
  sanitizer_stacktrace.cc
  sanitizer_stacktrace_libcdep.cc
  sanitizer_stacktrace_printer.cc
  sanitizer_stacktrace_sparc.cc
  sanitizer_symbolizer.cc
  sanitizer_symbolizer_libbacktrace.cc
  sanitizer_symbolizer_libcdep.cc
  sanitizer_symbolizer_mac.cc
  sanitizer_symbolizer_markup.cc
  sanitizer_symbolizer_posix_libcdep.cc
  sanitizer_symbolizer_report.cc
  sanitizer_symbolizer_win.cc
  sanitizer_unwind_linux_libcdep.cc
  sanitizer_unwind_win.cc)

# Explicitly list all sanitizer_common headers. Not all of these are
# included in sanitizer_common source files, but we need to depend on
# headers when building our custom unit tests.
set(SANITIZER_HEADERS
  sanitizer_addrhashmap.h
  sanitizer_allocator.h
  sanitizer_allocator_bytemap.h
  sanitizer_allocator_combined.h
  sanitizer_allocator_interface.h
  sanitizer_allocator_internal.h
  sanitizer_allocator_local_cache.h
  sanitizer_allocator_primary32.h
  sanitizer_allocator_primary64.h
  sanitizer_allocator_secondary.h
  sanitizer_allocator_size_class_map.h
  sanitizer_allocator_stats.h
  sanitizer_atomic.h
  sanitizer_atomic_clang.h
  sanitizer_atomic_msvc.h
  sanitizer_bitvector.h
  sanitizer_bvgraph.h
  sanitizer_common.h
  sanitizer_common_interceptors.inc
  sanitizer_common_interceptors_ioctl.inc
  sanitizer_common_interceptors_format.inc
  sanitizer_common_syscalls.inc
  sanitizer_deadlock_detector.h
  sanitizer_deadlock_detector_interface.h
  sanitizer_errno.h
  sanitizer_errno_codes.h
  sanitizer_file.h
  sanitizer_flag_parser.h
  sanitizer_flags.h
  sanitizer_flags.inc
  sanitizer_fuchsia.h
  sanitizer_interceptors_ioctl_netbsd.inc
  sanitizer_interface_internal.h
  sanitizer_internal_defs.h
  sanitizer_lfstack.h
  sanitizer_libc.h
  sanitizer_libignore.h
  sanitizer_linux.h
  sanitizer_list.h
  sanitizer_mac.h
  sanitizer_mutex.h
  sanitizer_persistent_allocator.h
  sanitizer_placement_new.h
  sanitizer_platform.h
  sanitizer_platform_interceptors.h
  sanitizer_platform_limits_netbsd.h
  sanitizer_platform_limits_posix.h
  sanitizer_platform_limits_solaris.h
  sanitizer_posix.h
  sanitizer_procmaps.h
  sanitizer_quarantine.h
  sanitizer_report_decorator.h
  sanitizer_stackdepot.h
  sanitizer_stackdepotbase.h
  sanitizer_stacktrace.h
  sanitizer_stacktrace_printer.h
  sanitizer_stoptheworld.h
  sanitizer_suppressions.h
  sanitizer_symbolizer.h
  sanitizer_symbolizer_fuchsia.h
  sanitizer_symbolizer_internal.h
  sanitizer_symbolizer_libbacktrace.h
  sanitizer_symbolizer_mac.h
  sanitizer_symbolizer_rtems.h
  sanitizer_syscall_generic.inc
  sanitizer_syscall_linux_x86_64.inc
  sanitizer_syscall_linux_aarch64.inc
  sanitizer_syscalls_netbsd.inc
  sanitizer_thread_registry.h
  sanitizer_vector.h
  sanitizer_win.h)

include_directories(..)

set(SANITIZER_COMMON_DEFINITIONS)

include(CheckIncludeFile)
append_have_file_definition(rpc/xdr.h HAVE_RPC_XDR_H SANITIZER_COMMON_DEFINITIONS)
append_have_file_definition(tirpc/rpc/xdr.h HAVE_TIRPC_RPC_XDR_H SANITIZER_COMMON_DEFINITIONS)

set(SANITIZER_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF SANITIZER_CFLAGS)

append_list_if(SANITIZER_LIMIT_FRAME_SIZE -Wframe-larger-than=570
               SANITIZER_CFLAGS)
append_list_if(COMPILER_RT_HAS_WGLOBAL_CONSTRUCTORS_FLAG -Wglobal-constructors
               SANITIZER_CFLAGS)

if (LLVM_ENABLE_PEDANTIC AND UNIX AND NOT APPLE)
  # With -pedantic, our .S files raise warnings about empty macro arguments
  # from __USER_LABEL_PREFIX__ being an empty arg to GLUE().  Unfortunately,
  # there is no simple way to test for an empty define, nor to disable just
  # that warning or to disable -pedantic.  There is also no simple way to
  # remove -pedantic from just this file (we'd have to remove from
  # CMAKE_C*_FLAGS and re-add as a source property to all the non-.S files).
  set_source_files_properties(sanitizer_linux_x86_64.S
    PROPERTIES COMPILE_FLAGS "-w")
  set_source_files_properties(sanitizer_linux_mips64.S
    PROPERTIES COMPILE_FLAGS "-w")
endif ()

if(APPLE)
  set(OS_OPTION OS ${SANITIZER_COMMON_SUPPORTED_OS})
endif()

add_compiler_rt_object_libraries(RTSanitizerCommon
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANITIZER_SOURCES}
  CFLAGS ${SANITIZER_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})
add_compiler_rt_object_libraries(RTSanitizerCommonNoTermination
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANITIZER_SOURCES_NOTERMINATION}
  CFLAGS ${SANITIZER_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})
add_compiler_rt_object_libraries(RTSanitizerCommonNoLibc
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANITIZER_NOLIBC_SOURCES}
  CFLAGS ${SANITIZER_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})
add_compiler_rt_object_libraries(RTSanitizerCommonLibc
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANITIZER_LIBCDEP_SOURCES}
  CFLAGS ${SANITIZER_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})
add_compiler_rt_object_libraries(RTSanitizerCommonCoverage
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANITIZER_COVERAGE_SOURCES}
  CFLAGS ${SANITIZER_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})
add_compiler_rt_object_libraries(RTSanitizerCommonSymbolizer
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANITIZER_SYMBOLIZER_SOURCES}
  CFLAGS ${SANITIZER_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})

set(SANITIZER_NO_WEAK_HOOKS_CFLAGS ${SANITIZER_CFLAGS})
list(APPEND SANITIZER_NO_WEAK_HOOKS_CFLAGS "-DSANITIZER_SUPPORTS_WEAK_HOOKS=0")
add_compiler_rt_object_libraries(RTSanitizerCommonNoHooks
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANITIZER_SOURCES}
  CFLAGS ${SANITIZER_NO_WEAK_HOOKS_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})
add_compiler_rt_object_libraries(RTSanitizerCommonLibcNoHooks
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANITIZER_LIBCDEP_SOURCES}
  CFLAGS ${SANITIZER_NO_WEAK_HOOKS_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})
add_compiler_rt_object_libraries(RTSanitizerCommonSymbolizerNoHooks
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANITIZER_SYMBOLIZER_SOURCES}
  CFLAGS ${SANITIZER_NO_WEAK_HOOKS_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})

if(OS_NAME MATCHES "SunOS")
  # Solaris ld doesn't support the non-standard GNU ld extension of adding
  # __start_SECNAME and __stop_SECNAME labels to sections whose names are
  # valid C identifiers.  Instead we add our own definitions for the
  # __sancov_guards section.
  add_compiler_rt_object_libraries(SancovBegin
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES sancov_begin.S
    CFLAGS ${SANITIZER_CFLAGS}
    DEFS ${SANITIZER_COMMON_DEFINITIONS})

  add_compiler_rt_runtime(clang_rt.sancov_begin
    STATIC
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    OBJECT_LIBS SancovBegin
    CFLAGS ${SANITIZER_CFLAGS}
    DEFS ${SANITIZER_COMMON_DEFINITIONS})

  add_compiler_rt_object_libraries(SancovEnd
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES sancov_end.S
    CFLAGS ${SANITIZER_CFLAGS}
    DEFS ${SANITIZER_COMMON_DEFINITIONS})

  add_compiler_rt_runtime(clang_rt.sancov_end
    STATIC
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    OBJECT_LIBS SancovEnd
    CFLAGS ${SANITIZER_CFLAGS}
    DEFS ${SANITIZER_COMMON_DEFINITIONS})
endif()

if(WIN32)
  add_compiler_rt_object_libraries(SanitizerCommonWeakInterception
    ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES sanitizer_win_weak_interception.cc
    CFLAGS ${SANITIZER_CFLAGS} -DSANITIZER_DYNAMIC
    DEFS ${SANITIZER_COMMON_DEFINITIONS})
  add_compiler_rt_object_libraries(SancovWeakInterception
    ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES sanitizer_coverage_win_weak_interception.cc
    CFLAGS ${SANITIZER_CFLAGS} -DSANITIZER_DYNAMIC
    DEFS ${SANITIZER_COMMON_DEFINITIONS})

  add_compiler_rt_object_libraries(SanitizerCommonDllThunk
    ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES sanitizer_win_dll_thunk.cc
    CFLAGS ${SANITIZER_CFLAGS} -DSANITIZER_DLL_THUNK
    DEFS ${SANITIZER_COMMON_DEFINITIONS})
  add_compiler_rt_object_libraries(SancovDllThunk
    ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES sanitizer_coverage_win_dll_thunk.cc
            sanitizer_coverage_win_sections.cc
    CFLAGS ${SANITIZER_CFLAGS} -DSANITIZER_DLL_THUNK
    DEFS ${SANITIZER_COMMON_DEFINITIONS})

  set(DYNAMIC_RUNTIME_THUNK_CFLAGS "-DSANITIZER_DYNAMIC_RUNTIME_THUNK")
  if(MSVC)
    list(APPEND DYNAMIC_RUNTIME_THUNK_CFLAGS "-Zl")
  elseif(CMAKE_C_COMPILER_ID MATCHES Clang)
    list(APPEND DYNAMIC_RUNTIME_THUNK_CFLAGS "-nodefaultlibs")
  endif()
  add_compiler_rt_object_libraries(SanitizerCommonDynamicRuntimeThunk
    ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES sanitizer_win_dynamic_runtime_thunk.cc
    CFLAGS ${SANITIZER_CFLAGS} ${DYNAMIC_RUNTIME_THUNK_CFLAGS}
    DEFS ${SANITIZER_COMMON_DEFINITIONS})
  add_compiler_rt_object_libraries(SancovDynamicRuntimeThunk
    ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES sanitizer_coverage_win_dynamic_runtime_thunk.cc
            sanitizer_coverage_win_sections.cc
    CFLAGS ${SANITIZER_CFLAGS} ${DYNAMIC_RUNTIME_THUNK_CFLAGS}
    DEFS ${SANITIZER_COMMON_DEFINITIONS})
endif()

# Unit tests for common sanitizer runtime.
if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
