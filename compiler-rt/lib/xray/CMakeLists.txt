# Build for all components of the XRay runtime support library.

# XRay runtime library implementation files.
set(XRAY_SOURCES
  xray_init.cc
  xray_flags.cc
  xray_interface.cc
  xray_log_interface.cc
  xray_utils.cc)

# Implementation files for all XRay modes.
set(XRAY_FDR_MODE_SOURCES
  xray_buffer_queue.cc
  xray_fdr_logging.cc)

set(XRAY_BASIC_MODE_SOURCES
  xray_inmemory_log.cc)

# Implementation files for all XRay architectures.
set(aarch64_SOURCES xray_AArch64.cc xray_trampoline_AArch64.S)
set(arm_SOURCES xray_arm.cc xray_trampoline_arm.S)
set(armhf_SOURCES ${arm_SOURCES})
set(mips_SOURCES xray_mips.cc xray_trampoline_mips.S)
set(mipsel_SOURCES xray_mips.cc xray_trampoline_mips.S)
set(mips64_SOURCES xray_mips64.cc xray_trampoline_mips64.S)
set(mips64el_SOURCES xray_mips64.cc xray_trampoline_mips64.S)
set(powerpc64le_SOURCES
    xray_powerpc64.cc
    xray_trampoline_powerpc64.cc
    xray_trampoline_powerpc64_asm.S)
set(x86_64_SOURCES xray_x86_64.cc xray_trampoline_x86_64.S)

# Now put it all together...
include_directories(..)
include_directories(../../include)

set(XRAY_CFLAGS ${SANITIZER_COMMON_CFLAGS})
set(XRAY_COMMON_DEFINITIONS XRAY_HAS_EXCEPTIONS=1)
append_list_if(
  COMPILER_RT_HAS_XRAY_COMPILER_FLAG XRAY_SUPPORTED=1 XRAY_COMMON_DEFINITIONS)
append_list_if(
  COMPILER_RT_BUILD_XRAY_NO_PREINIT XRAY_NO_PREINIT XRAY_COMMON_DEFINITIONS)

add_compiler_rt_component(xray)

set(XRAY_COMMON_RUNTIME_OBJECT_LIBS
    RTSanitizerCommon
    RTSanitizerCommonLibc)

if (APPLE)
  set(XRAY_LINK_LIBS ${SANITIZER_COMMON_LINK_LIBS})
  add_asm_sources(XRAY_ASM_SOURCES xray_trampoline_x86_64.S)

  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)
  add_weak_symbols("xray" WEAK_SYMBOL_LINK_FLAGS)

  add_compiler_rt_object_libraries(RTXray
    OS ${XRAY_SUPPORTED_OS}
    ARCHS ${XRAY_SUPPORTED_ARCH}
    SOURCES ${x86_64_SOURCES}
    CFLAGS ${XRAY_CFLAGS}
    DEFS ${XRAY_COMMON_DEFINITIONS})
  add_compiler_rt_object_libraries(RTXrayFDR
    OS ${XRAY_SUPPORTED_OS}
    ARCHS ${XRAY_SUPPORTED_ARCH}
    SOURCES ${XRAY_FDR_MODE_SOURCES}
    CFLAGS ${XRAY_CFLAGS}
    DEFS ${XRAY_COMMON_DEFINITIONS})
  add_compiler_rt_object_libraries(RTXrayBASIC
    OS ${XRAY_SUPPORTED_OS}
    ARCHS ${XRAY_SUPPORTED_ARCH}
    SOURCES ${XRAY_BASIC_MODE_SOURCES}
    CFLAGS ${XRAY_CFLAGS}
    DEFS ${XRAY_COMMON_DEFINITIONS})

  # We only support running on osx for now.
  add_compiler_rt_runtime(clang_rt.xray
    STATIC
    OS ${XRAY_SUPPORTED_OS}
    ARCHS ${XRAY_SUPPORTED_ARCH}
    OBJECT_LIBS RTXray
                RTSanitizerCommon
                RTSanitizerCommonLibc
    CFLAGS ${XRAY_CFLAGS}
    DEFS ${XRAY_COMMON_DEFINITIONS}
    LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS}
    LINK_LIBS ${XRAY_LINK_LIBS}
    PARENT_TARGET xray)
  add_compiler_rt_runtime(clang_rt.xray-fdr
    STATIC
    OS ${XRAY_SUPPORTED_OS}
    ARCHS ${XRAY_SUPPORTED_ARCH}
    OBJECT_LIBS RTXrayFDR
    CFLAGS ${XRAY_CFLAGS}
    DEFS ${XRAY_COMMON_DEFINITIONS}
    LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS}
    LINK_LIBS ${XRAY_LINK_LIBS}
    PARENT_TARGET xray)
  add_compiler_rt_runtime(clang_rt.xray-basic
    STATIC
    OS ${XRAY_SUPPORTED_OS}
    ARCHS ${XRAY_SUPPORTED_ARCH}
    OBJECT_LIBS RTXrayBASIC
    CFLAGS ${XRAY_CFLAGS}
    DEFS ${XRAY_COMMON_DEFINITIONS}
    LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS}
    LINK_LIBS ${XRAY_LINK_LIBS}
    PARENT_TARGET xray)
else() # not Apple
  foreach(arch ${XRAY_SUPPORTED_ARCH})
    if(NOT CAN_TARGET_${arch})
      continue()
    endif()
    add_compiler_rt_object_libraries(RTXray
      ARCHS ${arch}
      SOURCES ${XRAY_SOURCES} ${${arch}_SOURCES} CFLAGS ${XRAY_CFLAGS}
      DEFS ${XRAY_COMMON_DEFINITIONS})
    add_compiler_rt_object_libraries(RTXrayFDR
      ARCHS ${arch}
      SOURCES ${XRAY_FDR_MODE_SOURCES} CFLAGS ${XRAY_CFLAGS}
      DEFS ${XRAY_COMMON_DEFINITIONS})
    add_compiler_rt_object_libraries(RTXrayBASIC
      ARCHS ${arch}
      SOURCES ${XRAY_BASIC_MODE_SOURCES} CFLAGS ${XRAY_CFLAGS}
      DEFS ${XRAY_COMMON_DEFINITIONS})

    # Common XRay archive for instrumented binaries.
    add_compiler_rt_runtime(clang_rt.xray
     STATIC
     ARCHS ${arch}
     CFLAGS ${XRAY_CFLAGS}
     DEFS ${XRAY_COMMON_DEFINITIONS}
     OBJECT_LIBS ${XRAY_COMMON_RUNTIME_OBJECT_LIBS} RTXray
     PARENT_TARGET xray)
    # FDR mode runtime archive (addon for clang_rt.xray)
    add_compiler_rt_runtime(clang_rt.xray-fdr
      STATIC
      ARCHS ${arch}
      CFLAGS ${XRAY_CFLAGS}
      DEFS ${XRAY_COMMON_DEFINITIONS}
      OBJECT_LIBS RTXrayFDR
      PARENT_TARGET xray)
    # Basic mode runtime archive (addon for clang_rt.xray)
    add_compiler_rt_runtime(clang_rt.xray-basic
      STATIC
      ARCHS ${arch}
      CFLAGS ${XRAY_CFLAGS}
      DEFS ${XRAY_COMMON_DEFINITIONS}
      OBJECT_LIBS RTXrayBASIC
      PARENT_TARGET xray)
  endforeach()
endif() # not Apple

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
