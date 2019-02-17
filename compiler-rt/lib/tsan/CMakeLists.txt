# Build for the ThreadSanitizer runtime support library.

include_directories(..)

set(TSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
# SANITIZER_COMMON_CFLAGS contains -fPIC, but it's performance-critical for
# TSan runtime to be built with -fPIE to reduce the number of register spills.
# On FreeBSD however it provokes linkage issue thus we disable it.
if(NOT CMAKE_SYSTEM MATCHES "FreeBSD")
  append_list_if(COMPILER_RT_HAS_FPIE_FLAG -fPIE TSAN_CFLAGS)
endif()
append_rtti_flag(OFF TSAN_CFLAGS)

if(COMPILER_RT_TSAN_DEBUG_OUTPUT)
  # Add extra debug information to TSan runtime. This configuration is rarely
  # used, but we need to support it so that debug output will not bitrot.
  list(APPEND TSAN_CFLAGS -DTSAN_COLLECT_STATS=1
                          -DTSAN_DEBUG_OUTPUT=2)
endif()

set(TSAN_RTL_CFLAGS ${TSAN_CFLAGS})
append_list_if(COMPILER_RT_HAS_MSSE3_FLAG -msse3 TSAN_RTL_CFLAGS)
append_list_if(SANITIZER_LIMIT_FRAME_SIZE -Wframe-larger-than=530
               TSAN_RTL_CFLAGS)
append_list_if(COMPILER_RT_HAS_WGLOBAL_CONSTRUCTORS_FLAG -Wglobal-constructors
               TSAN_RTL_CFLAGS)

set(TSAN_SOURCES
  rtl/tsan_clock.cc
  rtl/tsan_debugging.cc
  rtl/tsan_external.cc
  rtl/tsan_fd.cc
  rtl/tsan_flags.cc
  rtl/tsan_ignoreset.cc
  rtl/tsan_interceptors.cc
  rtl/tsan_interface.cc
  rtl/tsan_interface_ann.cc
  rtl/tsan_interface_atomic.cc
  rtl/tsan_interface_java.cc
  rtl/tsan_malloc_mac.cc
  rtl/tsan_md5.cc
  rtl/tsan_mman.cc
  rtl/tsan_mutex.cc
  rtl/tsan_mutexset.cc
  rtl/tsan_preinit.cc
  rtl/tsan_report.cc
  rtl/tsan_rtl.cc
  rtl/tsan_rtl_mutex.cc
  rtl/tsan_rtl_proc.cc
  rtl/tsan_rtl_report.cc
  rtl/tsan_rtl_thread.cc
  rtl/tsan_stack_trace.cc
  rtl/tsan_stat.cc
  rtl/tsan_suppressions.cc
  rtl/tsan_symbolize.cc
  rtl/tsan_sync.cc)

set(TSAN_CXX_SOURCES
  rtl/tsan_new_delete.cc)

if(APPLE)
  list(APPEND TSAN_SOURCES
    rtl/tsan_interceptors_mac.cc
    rtl/tsan_libdispatch_mac.cc
    rtl/tsan_platform_mac.cc
    rtl/tsan_platform_posix.cc)
elseif(UNIX)
  # Assume Linux
  list(APPEND TSAN_SOURCES
    rtl/tsan_platform_linux.cc
    rtl/tsan_platform_posix.cc)
endif()

set(TSAN_HEADERS
  rtl/tsan_clock.h
  rtl/tsan_defs.h
  rtl/tsan_dense_alloc.h
  rtl/tsan_fd.h
  rtl/tsan_flags.h
  rtl/tsan_flags.inc
  rtl/tsan_ignoreset.h
  rtl/tsan_interceptors.h
  rtl/tsan_interface.h
  rtl/tsan_interface_ann.h
  rtl/tsan_interface_inl.h
  rtl/tsan_interface_java.h
  rtl/tsan_mman.h
  rtl/tsan_mutex.h
  rtl/tsan_mutexset.h
  rtl/tsan_platform.h
  rtl/tsan_ppc_regs.h
  rtl/tsan_report.h
  rtl/tsan_rtl.h
  rtl/tsan_stack_trace.h
  rtl/tsan_stat.h
  rtl/tsan_suppressions.h
  rtl/tsan_symbolize.h
  rtl/tsan_sync.h
  rtl/tsan_trace.h
  rtl/tsan_update_shadow_word_inl.h)

set(TSAN_RUNTIME_LIBRARIES)
add_compiler_rt_component(tsan)

if(APPLE)
  add_asm_sources(TSAN_ASM_SOURCES rtl/tsan_rtl_amd64.S rtl/tsan_rtl_aarch64.S)

  set(TSAN_LINK_LIBS ${SANITIZER_COMMON_LINK_LIBS})

  add_weak_symbols("ubsan" WEAK_SYMBOL_LINK_FLAGS)
  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)

  add_compiler_rt_runtime(clang_rt.tsan
    SHARED
    OS ${TSAN_SUPPORTED_OS}
    ARCHS ${TSAN_SUPPORTED_ARCH}
    SOURCES ${TSAN_SOURCES} ${TSAN_CXX_SOURCES} ${TSAN_ASM_SOURCES}
    ADDITIONAL_HEADERS ${TSAN_HEADERS}
    OBJECT_LIBS RTInterception
                RTSanitizerCommon
                RTSanitizerCommonLibc
                RTSanitizerCommonCoverage
                RTSanitizerCommonSymbolizer
                RTUbsan
    CFLAGS ${TSAN_RTL_CFLAGS}
    LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS}
    LINK_LIBS ${TSAN_LINK_LIBS} objc
    PARENT_TARGET tsan)
  add_compiler_rt_object_libraries(RTTsan_dynamic
    OS ${TSAN_SUPPORTED_OS}
    ARCHS ${TSAN_SUPPORTED_ARCH}
    SOURCES ${TSAN_SOURCES} ${TSAN_CXX_SOURCES} ${TSAN_ASM_SOURCES}
    ADDITIONAL_HEADERS ${TSAN_HEADERS}
    CFLAGS ${TSAN_RTL_CFLAGS})

  # Build and check Go runtime.
  set(BUILDGO_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/go/buildgo.sh)
  add_custom_target(GotsanRuntimeCheck
    COMMAND env "CC=${CMAKE_C_COMPILER} ${OSX_SYSROOT_FLAG}"
            IN_TMPDIR=1 SILENT=1 ${BUILDGO_SCRIPT}
    DEPENDS tsan ${BUILDGO_SCRIPT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/go
    COMMENT "Checking TSan Go runtime..."
    VERBATIM)
  set_target_properties(GotsanRuntimeCheck PROPERTIES FOLDER "Compiler-RT Misc")
else()
  foreach(arch ${TSAN_SUPPORTED_ARCH})
    if(arch STREQUAL "x86_64")
      add_asm_sources(TSAN_ASM_SOURCES rtl/tsan_rtl_amd64.S)
      # Sanity check for Go runtime.
      set(BUILDGO_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/go/buildgo.sh)
      add_custom_target(GotsanRuntimeCheck
        COMMAND env "CC=${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1}"
                IN_TMPDIR=1 SILENT=1 ${BUILDGO_SCRIPT}
        DEPENDS clang_rt.tsan-${arch} ${BUILDGO_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/go
        COMMENT "Checking TSan Go runtime..."
        VERBATIM)
    elseif(arch STREQUAL "aarch64")
      add_asm_sources(TSAN_ASM_SOURCES rtl/tsan_rtl_aarch64.S)
      # Sanity check for Go runtime.
      set(BUILDGO_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/go/buildgo.sh)
      add_custom_target(GotsanRuntimeCheck
	COMMAND env "CC=${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1}"
		IN_TMPDIR=1 SILENT=1 ${BUILDGO_SCRIPT}
	DEPENDS clang_rt.tsan-${arch} ${BUILDGO_SCRIPT}
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/go
	COMMENT "Checking TSan Go runtime..."
	VERBATIM)
    elseif(arch MATCHES "powerpc64|powerpc64le")
      add_asm_sources(TSAN_ASM_SOURCES rtl/tsan_rtl_ppc64.S)
      # Sanity check for Go runtime.
      set(BUILDGO_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/go/buildgo.sh)
      add_custom_target(GotsanRuntimeCheck
	COMMAND env "CC=${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1}"
		IN_TMPDIR=1 SILENT=1 ${BUILDGO_SCRIPT}
	DEPENDS clang_rt.tsan-${arch} ${BUILDGO_SCRIPT}
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/go
	COMMENT "Checking TSan Go runtime..."
	VERBATIM)
    elseif(arch MATCHES "mips64|mips64le")
      add_asm_sources(TSAN_ASM_SOURCES rtl/tsan_rtl_mips64.S)
    else()
      set(TSAN_ASM_SOURCES)
    endif()
    add_compiler_rt_runtime(clang_rt.tsan
      STATIC
      ARCHS ${arch}
      SOURCES ${TSAN_SOURCES} ${TSAN_ASM_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
              $<TARGET_OBJECTS:RTUbsan.${arch}>
      ADDITIONAL_HEADERS ${TSAN_HEADERS}
      CFLAGS ${TSAN_RTL_CFLAGS}
      PARENT_TARGET tsan)
    add_compiler_rt_runtime(clang_rt.tsan_cxx
      STATIC
      ARCHS ${arch}
      SOURCES ${TSAN_CXX_SOURCES}
              $<TARGET_OBJECTS:RTUbsan_cxx.${arch}>
      ADDITIONAL_HEADERS ${TSAN_HEADERS}
      CFLAGS ${TSAN_RTL_CFLAGS}
      PARENT_TARGET tsan)
    list(APPEND TSAN_RUNTIME_LIBRARIES clang_rt.tsan-${arch}
                                       clang_rt.tsan_cxx-${arch})
    add_sanitizer_rt_symbols(clang_rt.tsan
      ARCHS ${arch}
      EXTRA rtl/tsan.syms.extra)
    add_sanitizer_rt_symbols(clang_rt.tsan_cxx
      ARCHS ${arch}
      EXTRA rtl/tsan.syms.extra)
    add_dependencies(tsan clang_rt.tsan-${arch}
                          clang_rt.tsan_cxx-${arch}
                          clang_rt.tsan-${arch}-symbols
                          clang_rt.tsan_cxx-${arch}-symbols)
  endforeach()
endif()

# Make sure that non-platform-specific files don't include any system headers.
# FreeBSD/NetBSD do not install a number of Clang-provided headers for the
# compiler in the base system due to incompatibilities between FreeBSD/NetBSD's
# and Clang's versions. As a workaround do not use --sysroot=. on FreeBSD/NetBSD
# until this is addressed.
if(COMPILER_RT_HAS_SYSROOT_FLAG AND NOT CMAKE_SYSTEM_NAME MATCHES "FreeBSD"
   AND NOT CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  file(GLOB _tsan_generic_sources rtl/tsan*)
  file(GLOB _tsan_platform_sources rtl/tsan*posix* rtl/tsan*mac*
                                   rtl/tsan*linux*)
  list(REMOVE_ITEM _tsan_generic_sources ${_tsan_platform_sources})
  set_source_files_properties(${_tsan_generic_sources}
    PROPERTIES COMPILE_FLAGS "--sysroot=.")
endif()

# Build libcxx instrumented with TSan.
if(COMPILER_RT_LIBCXX_PATH AND
   COMPILER_RT_LIBCXXABI_PATH AND
   COMPILER_RT_TEST_COMPILER_ID STREQUAL "Clang" AND
   NOT ANDROID)
  set(libcxx_tsan_deps)
  foreach(arch ${TSAN_SUPPORTED_ARCH})
    get_target_flags_for_arch(${arch} TARGET_CFLAGS)
    set(LIBCXX_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libcxx_tsan_${arch})
    add_custom_libcxx(libcxx_tsan_${arch} ${LIBCXX_PREFIX}
      DEPS ${TSAN_RUNTIME_LIBRARIES}
      CFLAGS ${TARGET_CFLAGS} -fsanitize=thread
      USE_TOOLCHAIN)
    list(APPEND libcxx_tsan_deps libcxx_tsan_${arch}-build)
  endforeach()

  add_custom_target(libcxx_tsan DEPENDS ${libcxx_tsan_deps})
  set_target_properties(libcxx_tsan PROPERTIES FOLDER "Compiler-RT Misc")
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
