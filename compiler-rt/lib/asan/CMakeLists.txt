# Build for the AddressSanitizer runtime support library.

set(ASAN_SOURCES
  asan_allocator.cc
  asan_activation.cc
  asan_debugging.cc
  asan_descriptions.cc
  asan_errors.cc
  asan_fake_stack.cc
  asan_flags.cc
  asan_globals.cc
  asan_interceptors.cc
  asan_linux.cc
  asan_mac.cc
  asan_malloc_linux.cc
  asan_malloc_mac.cc
  asan_malloc_win.cc
  asan_memory_profile.cc
  asan_poisoning.cc
  asan_posix.cc
  asan_report.cc
  asan_rtl.cc
  asan_stack.cc
  asan_stats.cc
  asan_suppressions.cc
  asan_thread.cc
  asan_win.cc)

set(ASAN_CXX_SOURCES
  asan_new_delete.cc)

set(ASAN_PREINIT_SOURCES
  asan_preinit.cc)

include_directories(..)

set(ASAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF ASAN_CFLAGS)

set(ASAN_DYNAMIC_LINK_FLAGS)

if(ANDROID)
  list(APPEND ASAN_COMMON_DEFINITIONS
    ASAN_LOW_MEMORY=1)
# On Android, -z global does not do what it is documented to do.
# On Android, -z global moves the library ahead in the lookup order,
# placing it right after the LD_PRELOADs. This is used to compensate for the fact
# that Android linker does not look at the dependencies of the main executable
# that aren't dependencies of the current DSO when resolving symbols from said DSO.
# As a net result, this allows running ASan executables without LD_PRELOAD-ing the
# ASan runtime library.
# The above is applicable to L MR1 or newer.
  if (COMPILER_RT_HAS_Z_GLOBAL)
    list(APPEND ASAN_DYNAMIC_LINK_FLAGS -Wl,-z,global)
  endif()
endif()

set(ASAN_DYNAMIC_DEFINITIONS
  ${ASAN_COMMON_DEFINITIONS} ASAN_DYNAMIC=1)
append_list_if(WIN32 INTERCEPTION_DYNAMIC_CRT ASAN_DYNAMIC_DEFINITIONS)

set(ASAN_DYNAMIC_CFLAGS ${ASAN_CFLAGS})
append_list_if(COMPILER_RT_HAS_FTLS_MODEL_INITIAL_EXEC
  -ftls-model=initial-exec ASAN_DYNAMIC_CFLAGS)
append_list_if(MSVC /DEBUG ASAN_DYNAMIC_LINK_FLAGS)

append_list_if(COMPILER_RT_HAS_LIBC c ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBDL dl ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBM m ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBSTDCXX stdc++ ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBLOG log ASAN_DYNAMIC_LIBS)

# Compile ASan sources into an object library.

add_compiler_rt_object_libraries(RTAsan_dynamic
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${ASAN_SUPPORTED_ARCH}
  SOURCES ${ASAN_SOURCES} ${ASAN_CXX_SOURCES}
  CFLAGS ${ASAN_DYNAMIC_CFLAGS}
  DEFS ${ASAN_DYNAMIC_DEFINITIONS})

if(NOT APPLE)
  add_compiler_rt_object_libraries(RTAsan
    ARCHS ${ASAN_SUPPORTED_ARCH}
    SOURCES ${ASAN_SOURCES} CFLAGS ${ASAN_CFLAGS}
    DEFS ${ASAN_COMMON_DEFINITIONS})
  add_compiler_rt_object_libraries(RTAsan_cxx
    ARCHS ${ASAN_SUPPORTED_ARCH}
    SOURCES ${ASAN_CXX_SOURCES} CFLAGS ${ASAN_CFLAGS}
    DEFS ${ASAN_COMMON_DEFINITIONS})
  add_compiler_rt_object_libraries(RTAsan_preinit
    ARCHS ${ASAN_SUPPORTED_ARCH}
    SOURCES ${ASAN_PREINIT_SOURCES} CFLAGS ${ASAN_CFLAGS}
    DEFS ${ASAN_COMMON_DEFINITIONS})

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cc "")
  add_compiler_rt_object_libraries(RTAsan_dynamic_version_script_dummy
    ARCHS ${ASAN_SUPPORTED_ARCH}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/dummy.cc
    CFLAGS ${ASAN_DYNAMIC_CFLAGS}
    DEFS ${ASAN_DYNAMIC_DEFINITIONS})
endif()

# Build ASan runtimes shipped with Clang.
add_compiler_rt_component(asan)

if(APPLE)
  add_compiler_rt_runtime(clang_rt.asan
    SHARED
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${ASAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTAsan_dynamic
                RTInterception
                RTSanitizerCommon
                RTSanitizerCommonLibc
                RTLSanCommon
                RTUbsan
    CFLAGS ${ASAN_DYNAMIC_CFLAGS}
    DEFS ${ASAN_DYNAMIC_DEFINITIONS}
    PARENT_TARGET asan)
else()
  # Build separate libraries for each target.

  set(ASAN_COMMON_RUNTIME_OBJECT_LIBS
    RTInterception
    RTSanitizerCommon
    RTSanitizerCommonLibc
    RTLSanCommon
    RTUbsan)

  add_compiler_rt_runtime(clang_rt.asan
    STATIC
    ARCHS ${ASAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTAsan_preinit
                RTAsan
                ${ASAN_COMMON_RUNTIME_OBJECT_LIBS}
    CFLAGS ${ASAN_CFLAGS}
    DEFS ${ASAN_COMMON_DEFINITIONS}
    PARENT_TARGET asan)

  add_compiler_rt_runtime(clang_rt.asan_cxx
    STATIC
    ARCHS ${ASAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTAsan_cxx
                RTUbsan_cxx
    CFLAGS ${ASAN_CFLAGS}
    DEFS ${ASAN_COMMON_DEFINITIONS}
    PARENT_TARGET asan)

  add_compiler_rt_runtime(clang_rt.asan-preinit
    STATIC
    ARCHS ${ASAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTAsan_preinit
    CFLAGS ${ASAN_CFLAGS}
    DEFS ${ASAN_COMMON_DEFINITIONS}
    PARENT_TARGET asan)

  foreach(arch ${ASAN_SUPPORTED_ARCH})
    if (UNIX AND NOT ${arch} MATCHES "i386|i686")
      add_sanitizer_rt_version_list(clang_rt.asan-dynamic-${arch}
                                    LIBS clang_rt.asan-${arch} clang_rt.asan_cxx-${arch}
                                    EXTRA asan.syms.extra)
      set(VERSION_SCRIPT_FLAG
           -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/clang_rt.asan-dynamic-${arch}.vers)
      set_source_files_properties(
        ${CMAKE_CURRENT_BINARY_DIR}/dummy.cc
        PROPERTIES
        OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/clang_rt.asan-dynamic-${arch}.vers)
    else()
      set(VERSION_SCRIPT_FLAG)
    endif()

    add_compiler_rt_runtime(clang_rt.asan
      SHARED
      ARCHS ${arch}
      OBJECT_LIBS ${ASAN_COMMON_RUNTIME_OBJECT_LIBS}
              RTAsan_dynamic
              # The only purpose of RTAsan_dynamic_version_script_dummy is to carry
              # a dependency of the shared runtime on the version script. With CMake
              # 3.1 or later it can be replaced with a straightforward
              # add_dependencies(clang_rt.asan-dynamic-${arch} clang_rt.asan-dynamic-${arch}-version-list)
              RTAsan_dynamic_version_script_dummy
              RTUbsan_cxx
      CFLAGS ${ASAN_DYNAMIC_CFLAGS}
      LINKFLAGS ${ASAN_DYNAMIC_LINK_FLAGS}
                ${VERSION_SCRIPT_FLAG}
      LINK_LIBS ${ASAN_DYNAMIC_LIBS}
      DEFS ${ASAN_DYNAMIC_DEFINITIONS}
      PARENT_TARGET asan)

    if (UNIX AND NOT ${arch} MATCHES "i386|i686")
      add_sanitizer_rt_symbols(clang_rt.asan_cxx
        ARCHS ${arch})
      add_dependencies(asan clang_rt.asan_cxx-${arch}-symbols)
      add_sanitizer_rt_symbols(clang_rt.asan
        ARCHS ${arch}
        EXTRA asan.syms.extra)
      add_dependencies(asan clang_rt.asan-${arch}-symbols)
    endif()

    if (WIN32)
      add_compiler_rt_runtime(clang_rt.asan_dll_thunk
        STATIC
        ARCHS ${arch}
        SOURCES asan_win_dll_thunk.cc
                $<TARGET_OBJECTS:RTInterception.${arch}>
        CFLAGS ${ASAN_CFLAGS} -DASAN_DLL_THUNK
        DEFS ${ASAN_COMMON_DEFINITIONS}
        PARENT_TARGET asan)

      set(DYNAMIC_RUNTIME_THUNK_CFLAGS "-DASAN_DYNAMIC_RUNTIME_THUNK")
      if(MSVC)
        list(APPEND DYNAMIC_RUNTIME_THUNK_CFLAGS "-Zl")
      elseif(CMAKE_C_COMPILER_ID MATCHES Clang)
        list(APPEND DYNAMIC_RUNTIME_THUNK_CFLAGS "-nodefaultlibs")
      endif()

      add_compiler_rt_runtime(clang_rt.asan_dynamic_runtime_thunk
        STATIC
        ARCHS ${arch}
        SOURCES asan_win_dynamic_runtime_thunk.cc
        CFLAGS ${ASAN_CFLAGS} ${DYNAMIC_RUNTIME_THUNK_CFLAGS}
        DEFS ${ASAN_COMMON_DEFINITIONS}
        PARENT_TARGET asan)
    endif()
  endforeach()
endif()

add_compiler_rt_resource_file(asan_blacklist asan_blacklist.txt asan)

add_subdirectory(scripts)

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
