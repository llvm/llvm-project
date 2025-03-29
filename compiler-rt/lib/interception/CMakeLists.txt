# Build for the runtime interception helper library.

set(INTERCEPTION_SOURCES
  interception_linux.cpp
  interception_mac.cpp
  interception_win.cpp
  interception_type_test.cpp
  )

set(INTERCEPTION_HEADERS
  interception.h
  interception_linux.h
  interception_mac.h
  interception_win.h
  )

if(MSVC)
  # asan on windows only supports the release dll version of the runtimes, in the interest of
  # only having one asan dll to support/test. Having asan statically linked
  # with the runtime might be possible, but it multiplies the number of scenerios to test.
  # the program USING sanitizers can use whatever version of the runtime it wants to.
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
endif()

include_directories(..)

set(INTERCEPTION_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF INTERCEPTION_CFLAGS)

# Silence warnings in system headers with MSVC.
if(NOT CLANG_CL)
  append_list_if(COMPILER_RT_HAS_EXTERNAL_FLAG "/experimental:external;/external:W0;/external:anglebrackets" INTERCEPTION_CFLAGS)
endif()

add_compiler_rt_object_libraries(RTInterception
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES ${INTERCEPTION_SOURCES}
    ADDITIONAL_HEADERS ${INTERCEPTION_HEADERS}
    CFLAGS ${INTERCEPTION_CFLAGS})

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
