# Build for the XRay runtime support library.

# XRay runtime library implementation files.
set(XRAY_SOURCES
  xray_inmemory_log.cc
  xray_init.cc
  xray_flags.cc
  xray_interface.cc
  xray_buffer_queue.cc
  xray_log_interface.cc
  xray_fdr_logging.cc
  xray_utils.cc)

set(x86_64_SOURCES
    xray_x86_64.cc
    xray_trampoline_x86_64.S
    ${XRAY_SOURCES})

set(arm_SOURCES
    xray_arm.cc
    xray_trampoline_arm.S
    ${XRAY_SOURCES})

set(armhf_SOURCES
  ${arm_SOURCES})

set(aarch64_SOURCES
  xray_AArch64.cc
  xray_trampoline_AArch64.S
  ${XRAY_SOURCES})

set(mips_SOURCES
    xray_mips.cc
    xray_trampoline_mips.S
    ${XRAY_SOURCES})

set(mipsel_SOURCES
    xray_mips.cc
    xray_trampoline_mips.S
    ${XRAY_SOURCES})

set(mips64_SOURCES
    xray_mips64.cc
    xray_trampoline_mips64.S
    ${XRAY_SOURCES})

set(mips64el_SOURCES
    xray_mips64.cc
    xray_trampoline_mips64.S
    ${XRAY_SOURCES})

set(powerpc64le_SOURCES
        xray_powerpc64.cc
        xray_trampoline_powerpc64.cc
        xray_trampoline_powerpc64_asm.S
        ${XRAY_SOURCES})

include_directories(..)
include_directories(../../include)

set(XRAY_CFLAGS ${SANITIZER_COMMON_CFLAGS})
set(XRAY_COMMON_DEFINITIONS XRAY_HAS_EXCEPTIONS=1)
append_list_if(
  COMPILER_RT_HAS_XRAY_COMPILER_FLAG XRAY_SUPPORTED=1 XRAY_COMMON_DEFINITIONS)

add_compiler_rt_object_libraries(RTXray
  ARCHS ${XRAY_SUPPORTED_ARCH}
  SOURCES ${XRAY_SOURCES} CFLAGS ${XRAY_CFLAGS}
  DEFS ${XRAY_COMMON_DEFINITIONS})

add_compiler_rt_component(xray)

set(XRAY_COMMON_RUNTIME_OBJECT_LIBS
    RTSanitizerCommon
    RTSanitizerCommonLibc)

foreach(arch ${XRAY_SUPPORTED_ARCH})
  if(CAN_TARGET_${arch})
    add_compiler_rt_runtime(clang_rt.xray
     STATIC
     ARCHS ${arch}
     SOURCES ${${arch}_SOURCES}
     CFLAGS ${XRAY_CFLAGS}
     DEFS ${XRAY_COMMON_DEFINITIONS}
     OBJECT_LIBS ${XRAY_COMMON_RUNTIME_OBJECT_LIBS}
     PARENT_TARGET xray)
  endif()
endforeach()

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
