# Build for the runtime interception helper library.

set(INTERCEPTION_SOURCES
  interception_linux.cc
  interception_mac.cc
  interception_win.cc
  interception_type_test.cc
  )

include_directories(..)

set(INTERCEPTION_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF INTERCEPTION_CFLAGS)

add_compiler_rt_object_libraries(RTInterception
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES ${INTERCEPTION_SOURCES}
    CFLAGS ${INTERCEPTION_CFLAGS})
