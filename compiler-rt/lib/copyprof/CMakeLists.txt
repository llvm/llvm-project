# Build for the CopyProf runtime support library.

set(COPYPROF_SOURCES
  copyprof.cpp
  copyprof_reporting.cpp
  copyprof_shadow.cpp
  copyprof_stack.cpp
  copyprof_state.cpp
)

set(COPYPROF_HEADERS
  copyprof_interface_internal.h
  copyprof_internal.h
  copyprof_reporting.h
  copyprof_shadow.h
  copyprof_state.h
)

include_directories(..)
include_directories(../../include)

set(COPYPROF_CFLAGS ${SANITIZER_COMMON_CFLAGS})
set(COPYPROF_COMMON_DEFINITIONS "")
append_rtti_flag(OFF COPYPROF_CFLAGS)

add_compiler_rt_object_libraries(RTCopyProf
  ARCHS ${COPYPROF_SUPPORTED_ARCH}
  SOURCES ${COPYPROF_SOURCES}
  ADDITIONAL_HEADERS ${COPYPROF_HEADERS}
  CFLAGS ${COPYPROF_CFLAGS}
  DEFS ${COPYPROF_COMMON_DEFINITIONS})

# Build CopyProf runtimes shipped with Clang.
add_compiler_rt_component(copyprof)

set(COPYPROF_COMMON_RUNTIME_OBJECT_LIBS
  RTInterception
  RTSanitizerCommon
  RTSanitizerCommonLibc
  RTSanitizerCommonSymbolizer
)

add_compiler_rt_runtime(clang_rt.copyprof
  STATIC
  ARCHS ${COPYPROF_SUPPORTED_ARCH}
  OBJECT_LIBS
  RTCopyProf
  ${COPYPROF_COMMON_RUNTIME_OBJECT_LIBS}
  CFLAGS ${COPYPROF_CFLAGS}
  DEFS ${COPYPROF_COMMON_DEFINITIONS}
  PARENT_TARGET copyprof)

if(SANITIZER_USE_SYMBOLS)
  foreach(arch ${COPYPROF_SUPPORTED_ARCH})
    add_sanitizer_rt_symbols(clang_rt.copyprof
      ARCHS ${arch}
      EXTRA copyprof.syms.extra)
    add_dependencies(copyprof clang_rt.copyprof-${arch}-symbols)
  endforeach()
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
