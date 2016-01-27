# Build for the undefined behavior sanitizer runtime support library.

set(UBSAN_SOURCES
  ubsan_diag.cc
  ubsan_init.cc
  ubsan_flags.cc
  ubsan_handlers.cc
  ubsan_value.cc
  )

set(UBSAN_STANDALONE_SOURCES
  ubsan_init_standalone.cc
  )

set(UBSAN_CXX_SOURCES
  ubsan_handlers_cxx.cc
  ubsan_type_hash.cc
  ubsan_type_hash_itanium.cc
  ubsan_type_hash_win.cc
  )

include_directories(..)

set(UBSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_no_rtti_flag(UBSAN_CFLAGS)
append_list_if(SANITIZER_CAN_USE_CXXABI -DUBSAN_CAN_USE_CXXABI UBSAN_CFLAGS)

set(UBSAN_STANDALONE_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_no_rtti_flag(UBSAN_STANDALONE_CFLAGS)
append_list_if(SANITIZER_CAN_USE_CXXABI -DUBSAN_CAN_USE_CXXABI UBSAN_STANDALONE_CFLAGS)

set(UBSAN_CXXFLAGS ${SANITIZER_COMMON_CFLAGS})
append_list_if(SANITIZER_CAN_USE_CXXABI -DUBSAN_CAN_USE_CXXABI UBSAN_CXXFLAGS)

add_custom_target(ubsan)

if(APPLE)
  set(UBSAN_COMMON_SOURCES ${UBSAN_SOURCES})
  if(SANITIZER_CAN_USE_CXXABI)
    list(APPEND UBSAN_COMMON_SOURCES ${UBSAN_CXX_SOURCES})
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

    add_compiler_rt_runtime(clang_rt.ubsan
      SHARED
      OS ${SANITIZER_COMMON_SUPPORTED_OS}
      ARCHS ${UBSAN_SUPPORTED_ARCH}
      OBJECT_LIBS RTUbsan
                  RTUbsan_standalone
                  RTSanitizerCommon
                  RTSanitizerCommonLibc
      PARENT_TARGET ubsan)
  endif()

else()
  # Common parts of UBSan runtime.
  add_compiler_rt_object_libraries(RTUbsan
    ARCHS ${UBSAN_COMMON_SUPPORTED_ARCH}
    SOURCES ${UBSAN_SOURCES} CFLAGS ${UBSAN_CFLAGS})
  # C++-specific parts of UBSan runtime. Requires a C++ ABI library.
  add_compiler_rt_object_libraries(RTUbsan_cxx
    ARCHS ${UBSAN_COMMON_SUPPORTED_ARCH}
    SOURCES ${UBSAN_CXX_SOURCES} CFLAGS ${UBSAN_CXXFLAGS})

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

add_dependencies(compiler-rt ubsan)
