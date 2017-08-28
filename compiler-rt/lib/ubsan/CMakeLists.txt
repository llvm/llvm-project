# Build for the undefined behavior sanitizer runtime support library.

set(UBSAN_SOURCES
  ubsan_diag.cc
  ubsan_init.cc
  ubsan_flags.cc
  ubsan_handlers.cc
  ubsan_value.cc
  )

set(UBSAN_STANDALONE_SOURCES
  ubsan_diag_standalone.cc
  ubsan_init_standalone.cc
  )

set(UBSAN_CXXABI_SOURCES
  ubsan_handlers_cxx.cc
  ubsan_type_hash.cc
  ubsan_type_hash_itanium.cc
  ubsan_type_hash_win.cc
  )

include_directories(..)

set(UBSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF UBSAN_CFLAGS)
append_list_if(SANITIZER_CAN_USE_CXXABI -DUBSAN_CAN_USE_CXXABI UBSAN_CFLAGS)

set(UBSAN_STANDALONE_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF UBSAN_STANDALONE_CFLAGS)
append_list_if(SANITIZER_CAN_USE_CXXABI -DUBSAN_CAN_USE_CXXABI UBSAN_STANDALONE_CFLAGS)

set(UBSAN_CXXFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(ON UBSAN_CXXFLAGS)
append_list_if(SANITIZER_CAN_USE_CXXABI -DUBSAN_CAN_USE_CXXABI UBSAN_CXXFLAGS)

set(UBSAN_DYNAMIC_LIBS ${SANITIZER_COMMON_LINK_LIBS})

append_list_if(COMPILER_RT_HAS_LIBDL dl UBSAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBLOG log UBSAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt UBSAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread UBSAN_DYNAMIC_LIBS)

list(APPEND UBSAN_DYNAMIC_LIBS ${SANITIZER_CXX_ABI_LIBRARY})

add_compiler_rt_component(ubsan)

if(APPLE)
  set(UBSAN_COMMON_SOURCES ${UBSAN_SOURCES})
  if(SANITIZER_CAN_USE_CXXABI)
    list(APPEND UBSAN_COMMON_SOURCES ${UBSAN_CXXABI_SOURCES})
  endif()

  # Common parts of UBSan runtime.
  add_compiler_rt_object_libraries(RTUbsan
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${UBSAN_COMMON_SUPPORTED_ARCH}
    SOURCES ${UBSAN_COMMON_SOURCES}
    CFLAGS ${UBSAN_CXXFLAGS})

  if(COMPILER_RT_HAS_UBSAN)
    # Initializer of standalone UBSan runtime.
    add_compiler_rt_object_libraries(RTUbsan_standalone
      OS ${SANITIZER_COMMON_SUPPORTED_OS}
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      SOURCES ${UBSAN_STANDALONE_SOURCES}
      CFLAGS ${UBSAN_STANDALONE_CFLAGS})

    add_weak_symbols("ubsan" WEAK_SYMBOL_LINK_FLAGS)
    add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)

    add_compiler_rt_runtime(clang_rt.ubsan
      SHARED
      OS ${SANITIZER_COMMON_SUPPORTED_OS}
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      OBJECT_LIBS RTUbsan
                  RTUbsan_standalone
                  RTSanitizerCommon
                  RTSanitizerCommonLibc
      LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS}
      PARENT_TARGET ubsan)
  endif()

else()
  # Common parts of UBSan runtime.
  add_compiler_rt_object_libraries(RTUbsan
    ARCHS ${UBSAN_COMMON_SUPPORTED_ARCH}
    SOURCES ${UBSAN_SOURCES} CFLAGS ${UBSAN_CFLAGS})

  if(SANITIZER_CAN_USE_CXXABI)
    # C++-specific parts of UBSan runtime. Requires a C++ ABI library.
    set(UBSAN_CXX_SOURCES ${UBSAN_CXXABI_SOURCES})
  else()
    # Dummy target if we don't have C++ ABI library.
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cxx_dummy.cc "")
    set(UBSAN_CXX_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/cxx_dummy.cc)
  endif()

  add_compiler_rt_object_libraries(RTUbsan_cxx
    ARCHS ${UBSAN_COMMON_SUPPORTED_ARCH}
    SOURCES ${UBSAN_CXX_SOURCES} CFLAGS ${UBSAN_CXXFLAGS})

  if (WIN32)
    add_compiler_rt_object_libraries(UbsanWeakInterception
      ${SANITIZER_COMMON_SUPPORTED_OS}
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      SOURCES ubsan_win_weak_interception.cc
      CFLAGS ${UBSAN_CFLAGS} -DSANITIZER_DYNAMIC
      DEFS ${UBSAN_COMMON_DEFINITIONS})

    add_compiler_rt_object_libraries(UbsanDllThunk
      ${SANITIZER_COMMON_SUPPORTED_OS}
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      SOURCES ubsan_win_dll_thunk.cc
      CFLAGS ${UBSAN_CFLAGS} -DSANITIZER_DLL_THUNK
      DEFS ${UBSAN_COMMON_DEFINITIONS})

    set(DYNAMIC_RUNTIME_THUNK_CFLAGS "-DSANITIZER_DYNAMIC_RUNTIME_THUNK")
    if(MSVC)
      list(APPEND DYNAMIC_RUNTIME_THUNK_CFLAGS "-Zl")
    elseif(CMAKE_C_COMPILER_ID MATCHES Clang)
      list(APPEND DYNAMIC_RUNTIME_THUNK_CFLAGS "-nodefaultlibs")
    endif()
    add_compiler_rt_object_libraries(UbsanDynamicRuntimeThunk
      ${SANITIZER_COMMON_SUPPORTED_OS}
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      SOURCES ubsan_win_dynamic_runtime_thunk.cc
      CFLAGS ${UBSAN_CFLAGS} ${DYNAMIC_RUNTIME_THUNK_CFLAGS}
      DEFS ${UBSAN_COMMON_DEFINITIONS})
  endif()

  if(COMPILER_RT_HAS_UBSAN)
    # Initializer of standalone UBSan runtime.
    add_compiler_rt_object_libraries(RTUbsan_standalone
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      SOURCES ${UBSAN_STANDALONE_SOURCES} CFLAGS ${UBSAN_STANDALONE_CFLAGS})

    # Standalone UBSan runtimes.
    add_compiler_rt_runtime(clang_rt.ubsan_standalone
      STATIC
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      OBJECT_LIBS RTSanitizerCommon
              RTSanitizerCommonLibc
              RTUbsan
              RTUbsan_standalone
      CFLAGS ${UBSAN_CFLAGS}
      PARENT_TARGET ubsan)

    add_compiler_rt_runtime(clang_rt.ubsan_standalone_cxx
      STATIC
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      OBJECT_LIBS RTUbsan_cxx
      CFLAGS ${UBSAN_CXXFLAGS}
      PARENT_TARGET ubsan)

    add_compiler_rt_runtime(clang_rt.ubsan_standalone
      SHARED
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      OBJECT_LIBS RTSanitizerCommon
              RTSanitizerCommonLibc
              RTUbsan
      CFLAGS ${UBSAN_CFLAGS}
      LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS}
      LINK_LIBS ${UBSAN_DYNAMIC_LIBS}
      PARENT_TARGET ubsan)

    add_compiler_rt_runtime(clang_rt.ubsan_standalone_cxx
      SHARED
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      OBJECT_LIBS RTSanitizerCommon
              RTSanitizerCommonLibc
              RTUbsan
              RTUbsan_cxx
      CFLAGS ${UBSAN_CXXFLAGS}
      LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS}
      LINK_LIBS ${UBSAN_DYNAMIC_LIBS}
      PARENT_TARGET ubsan)

    if (UNIX)
      set(ARCHS_FOR_SYMBOLS ${UBSAN_SUPPORTED_ARCH})
      list(REMOVE_ITEM ARCHS_FOR_SYMBOLS i386 i686)
      add_sanitizer_rt_symbols(clang_rt.ubsan_standalone
        ARCHS ${ARCHS_FOR_SYMBOLS}
        PARENT_TARGET ubsan
        EXTRA ubsan.syms.extra)
      add_sanitizer_rt_symbols(clang_rt.ubsan_standalone_cxx
        ARCHS ${ARCHS_FOR_SYMBOLS}
        PARENT_TARGET ubsan
        EXTRA ubsan.syms.extra)
    endif()
  endif()
endif()
