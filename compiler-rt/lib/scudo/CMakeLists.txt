add_compiler_rt_component(scudo)

include_directories(..)

set(SCUDO_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF SCUDO_CFLAGS)
list(APPEND SCUDO_CFLAGS -msse4.2 -mcx16)

set(SCUDO_SOURCES
  scudo_allocator.cpp
  scudo_flags.cpp
  scudo_interceptors.cpp
  scudo_new_delete.cpp
  scudo_termination.cpp
  scudo_utils.cpp)

if(COMPILER_RT_HAS_SCUDO)
  foreach(arch ${SCUDO_SUPPORTED_ARCH})
    add_compiler_rt_runtime(clang_rt.scudo
      STATIC
      ARCHS ${arch}
      SOURCES ${SCUDO_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonNoTermination.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
      CFLAGS ${SCUDO_CFLAGS}
      PARENT_TARGET scudo)
  endforeach()
endif()
