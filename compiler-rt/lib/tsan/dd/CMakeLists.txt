# Build for the experimental deadlock detector runtime library.

include_directories(../..)

set(DD_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF DD_CFLAGS)

set(DD_SOURCES
  dd_rtl.cc
  dd_interceptors.cc
)

set(DD_LINKLIBS)
append_list_if(COMPILER_RT_HAS_LIBDL dl DD_LINKLIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt DD_LINKLIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread DD_LINKLIBS)

add_custom_target(dd)
# Deadlock detector is currently supported on 64-bit Linux only.
if(CAN_TARGET_x86_64 AND UNIX AND NOT APPLE AND NOT ANDROID)
  set(arch "x86_64")
  add_compiler_rt_runtime(clang_rt.dd
    STATIC
    ARCHS ${arch}
    SOURCES ${DD_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    CFLAGS ${DD_CFLAGS}
    PARENT_TARGET dd)

  add_compiler_rt_object_libraries(RTDD
    ARCHS ${arch}
    SOURCES ${DD_SOURCES} CFLAGS ${DD_CFLAGS})

  add_compiler_rt_runtime(clang_rt.dyndd
    SHARED
    ARCHS ${arch}
    SOURCES $<TARGET_OBJECTS:RTDD.${arch}>
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    LINK_LIBS ${DD_LINKLIBS}
    PARENT_TARGET dd)
endif()

add_dependencies(compiler-rt dd)
