add_compiler_rt_component(scudo_standalone)

include_directories(../..)

set(SCUDO_CFLAGS)

list(APPEND SCUDO_CFLAGS
  -Wall
  -nostdinc++)

# Remove -stdlib= which is unused when passing -nostdinc++.
string(REGEX REPLACE "-stdlib=[a-zA-Z+]*" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

append_list_if(COMPILER_RT_HAS_FFREESTANDING_FLAG -ffreestanding SCUDO_CFLAGS)

append_list_if(COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG -fvisibility=hidden SCUDO_CFLAGS)

if(COMPILER_RT_DEBUG)
  list(APPEND SCUDO_CFLAGS -O0)
else()
  list(APPEND SCUDO_CFLAGS -O3)
endif()

set(SCUDO_LINK_FLAGS)

list(APPEND SCUDO_LINK_FLAGS -Wl,-z,defs,-z,now,-z,relro)

append_list_if(COMPILER_RT_HAS_NODEFAULTLIBS_FLAG -nodefaultlibs SCUDO_LINK_FLAGS)

if(ANDROID)
# Put the shared library in the global group. For more details, see
# android-changes-for-ndk-developers.md#changes-to-library-search-order
  append_list_if(COMPILER_RT_HAS_Z_GLOBAL -Wl,-z,global SCUDO_LINK_FLAGS)
endif()

set(SCUDO_SOURCES
  common.cc
  fuchsia.cc
  linux.cc)

set(SCUDO_HEADERS
  atomic_helpers.h
  bytemap.h
  internal_defs.h
  linux.h
  list.h
  mutex.h
  platform.h
  stats.h
  vector.h)

if(COMPILER_RT_HAS_SCUDO_STANDALONE)
  add_compiler_rt_object_libraries(RTScudoStandalone
    ARCHS ${SCUDO_STANDALONE_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    CFLAGS ${SCUDO_CFLAGS})

  add_compiler_rt_runtime(clang_rt.scudo_standalone
    STATIC
    ARCHS ${SCUDO_STANDALONE_SUPPORTED_ARCH}
    SOURCES ${SCUDO_SOURCES}
    ADDITIONAL_HEADERS ${SCUDO_HEADERS}
    CFLAGS ${SCUDO_CFLAGS}
    PARENT_TARGET scudo_standalone)

  if(COMPILER_RT_INCLUDE_TESTS)
    add_subdirectory(tests)
  endif()
endif()
