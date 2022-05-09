# Get sources
set(LIBCXXABI_SOURCES
  # C++ABI files
  cxa_aux_runtime.cpp
  cxa_default_handlers.cpp
  cxa_demangle.cpp
  cxa_exception_storage.cpp
  cxa_guard.cpp
  cxa_handlers.cpp
  cxa_vector.cpp
  cxa_virtual.cpp
  # C++ STL files
  stdlib_exception.cpp
  stdlib_stdexcept.cpp
  stdlib_typeinfo.cpp
  # Internal files
  abort_message.cpp
  fallback_malloc.cpp
  private_typeinfo.cpp
)

if (LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS)
  list(APPEND LIBCXXABI_SOURCES
    stdlib_new_delete.cpp
  )
endif()

if (LIBCXXABI_ENABLE_EXCEPTIONS)
  list(APPEND LIBCXXABI_SOURCES
    cxa_exception.cpp
    cxa_personality.cpp
  )
else()
  list(APPEND LIBCXXABI_SOURCES
    cxa_noexception.cpp
  )
endif()

if (LIBCXXABI_ENABLE_THREADS AND (UNIX OR FUCHSIA) AND NOT (APPLE OR CYGWIN)
    AND NOT (${CMAKE_SYSTEM_NAME} MATCHES "AIX"))
  list(APPEND LIBCXXABI_SOURCES
    cxa_thread_atexit.cpp
  )
endif()

set(LIBCXXABI_HEADERS
  ../include/cxxabi.h
)

# Add all the headers to the project for IDEs.
if (MSVC_IDE OR XCODE)
  # Force them all into the headers dir on MSVC, otherwise they end up at
  # project scope because they don't have extensions.
  if (MSVC_IDE)
    source_group("Header Files" FILES ${LIBCXXABI_HEADERS})
  endif()
endif()

# Some files depend on libc++ internals.
include_directories("${LIBCXXABI_LIBCXX_PATH}/src")

if (LIBCXXABI_HAS_CXA_THREAD_ATEXIT_IMPL)
  add_definitions(-DHAVE___CXA_THREAD_ATEXIT_IMPL)
endif()

if (LIBCXXABI_ENABLE_FORGIVING_DYNAMIC_CAST)
  add_definitions(-D_LIBCXXABI_FORGIVING_DYNAMIC_CAST)
endif()

if (APPLE)
  add_library_flags_if(LIBCXXABI_HAS_SYSTEM_LIB System)
else()
  if (LIBCXXABI_ENABLE_THREADS)
    add_library_flags_if(LIBCXXABI_HAS_PTHREAD_LIB pthread)
  endif()

  add_library_flags_if(LIBCXXABI_HAS_C_LIB c)
endif()

if (LIBCXXABI_USE_COMPILER_RT)
  find_compiler_rt_library(builtins LIBCXXABI_BUILTINS_LIBRARY)
  list(APPEND LIBCXXABI_SHARED_LIBRARIES "${LIBCXXABI_BUILTINS_LIBRARY}")
endif ()

if (LIBCXXABI_USE_LLVM_UNWINDER)
  # Prefer using the in-tree version of libunwind, either shared or static. If
  # none are found fall back to using -lunwind.
  # FIXME: Is it correct to prefer the static version of libunwind?
  if (NOT LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_SHARED_LIBRARY AND (TARGET unwind_shared OR HAVE_LIBUNWIND))
    list(APPEND LIBCXXABI_SHARED_LIBRARIES unwind_shared)
  elseif (LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_SHARED_LIBRARY AND (TARGET unwind_static OR HAVE_LIBUNWIND))
    list(APPEND LIBCXXABI_SHARED_LIBRARIES unwind_static)
  else()
    list(APPEND LIBCXXABI_SHARED_LIBRARIES unwind)
  endif()
  if (NOT LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY AND (TARGET unwind_shared OR HAVE_LIBUNWIND))
    list(APPEND LIBCXXABI_STATIC_LIBRARIES unwind_shared)
  elseif (LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY AND (TARGET unwind_static OR HAVE_LIBUNWIND))
      # We handle this by directly merging libunwind objects into libc++abi.
  else()
    list(APPEND LIBCXXABI_STATIC_LIBRARIES unwind)
  endif()
else()
  add_library_flags_if(LIBCXXABI_HAS_GCC_S_LIB gcc_s)
endif()
if (MINGW)
  # MINGW_LIBRARIES is defined in config-ix.cmake
  list(APPEND LIBCXXABI_LIBRARIES ${MINGW_LIBRARIES})
endif()
if (ANDROID AND ANDROID_PLATFORM_LEVEL LESS 21)
  list(APPEND LIBCXXABI_LIBRARIES android_support)
endif()

if (NOT LIBCXXABI_USE_COMPILER_RT)
  add_library_flags_if(LIBCXXABI_HAS_GCC_LIB gcc)
endif ()

# Setup flags.
if (CXX_SUPPORTS_NOSTDLIBXX_FLAG)
  add_link_flags_if_supported(-nostdlib++)
else()
  add_link_flags_if_supported(-nodefaultlibs)
endif()

if ( APPLE )
  if (LLVM_USE_SANITIZER)
    if (("${LLVM_USE_SANITIZER}" STREQUAL "Address") OR
        ("${LLVM_USE_SANITIZER}" STREQUAL "Address;Undefined") OR
        ("${LLVM_USE_SANITIZER}" STREQUAL "Undefined;Address"))
      set(LIBFILE "libclang_rt.asan_osx_dynamic.dylib")
    elseif("${LLVM_USE_SANITIZER}" STREQUAL "Undefined")
      set(LIBFILE "libclang_rt.ubsan_osx_dynamic.dylib")
    elseif("${LLVM_USE_SANITIZER}" STREQUAL "Thread")
      set(LIBFILE "libclang_rt.tsan_osx_dynamic.dylib")
    else()
      message(WARNING "LLVM_USE_SANITIZER=${LLVM_USE_SANITIZER} is not supported on OS X")
    endif()
    if (LIBFILE)
      find_compiler_rt_library(builtins LIBCXXABI_BUILTINS_LIBRARY)
      get_filename_component(LIBDIR "${LIBCXXABI_BUILTINS_LIBRARY}" DIRECTORY)
      if (NOT IS_DIRECTORY "${LIBDIR}")
        message(FATAL_ERROR "Cannot find compiler-rt directory on OS X required for LLVM_USE_SANITIZER")
      endif()
      set(LIBCXXABI_SANITIZER_LIBRARY "${LIBDIR}/${LIBFILE}")
      set(LIBCXXABI_SANITIZER_LIBRARY "${LIBCXXABI_SANITIZER_LIBRARY}" PARENT_SCOPE)
      message(STATUS "Manually linking compiler-rt library: ${LIBCXXABI_SANITIZER_LIBRARY}")
      add_library_flags("${LIBCXXABI_SANITIZER_LIBRARY}")
      add_link_flags("-Wl,-rpath,${LIBDIR}")
    endif()
  endif()

  # Make sure we link in CrashReporterClient if we find it -- it's used by
  # abort() on Apple platforms when building the system dylib.
  find_library(CrashReporterClient NAMES libCrashReporterClient.a
                                   PATHS "${CMAKE_OSX_SYSROOT}/usr/local/lib")
  if (CrashReporterClient)
    message(STATUS "Linking with CrashReporterClient at ${CrashReporterClient}")
    add_library_flags("${CrashReporterClient}")
  else()
    message(STATUS "Could not find CrashReporterClient, not linking against it")
  endif()
endif()

split_list(LIBCXXABI_COMPILE_FLAGS)
split_list(LIBCXXABI_LINK_FLAGS)

# FIXME: libc++abi.so will not link when modules are enabled because it depends
# on symbols defined in libc++.so which has not yet been built.
if (LLVM_ENABLE_MODULES)
  string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
endif()

if (NOT TARGET pstl::ParallelSTL)
  message(STATUS "Could not find ParallelSTL, libc++abi will not attempt to use it but the build may fail if the libc++ in use needs it to be available.")
endif()

# Build the shared library.
if (LIBCXXABI_ENABLE_SHARED)
  add_library(cxxabi_shared SHARED ${LIBCXXABI_SOURCES} ${LIBCXXABI_HEADERS})
  target_link_libraries(cxxabi_shared PRIVATE cxx-headers ${LIBCXXABI_SHARED_LIBRARIES} ${LIBCXXABI_LIBRARIES})
  target_link_libraries(cxxabi_shared PUBLIC cxxabi-headers)
  if (TARGET pstl::ParallelSTL)
    target_link_libraries(cxxabi_shared PUBLIC pstl::ParallelSTL)
  endif()
  set_target_properties(cxxabi_shared
    PROPERTIES
      CXX_EXTENSIONS OFF
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED OFF
      COMPILE_FLAGS "${LIBCXXABI_COMPILE_FLAGS}"
      LINK_FLAGS "${LIBCXXABI_LINK_FLAGS}"
      OUTPUT_NAME "c++abi"
      SOVERSION "1"
      VERSION "${LIBCXXABI_LIBRARY_VERSION}"
      DEFINE_SYMBOL ""
  )

  list(APPEND LIBCXXABI_BUILD_TARGETS "cxxabi_shared")
  if (LIBCXXABI_INSTALL_SHARED_LIBRARY)
    list(APPEND LIBCXXABI_INSTALL_TARGETS "cxxabi_shared")
  endif()

  add_library(cxxabi-reexports INTERFACE)

  # -exported_symbols_list is only available on Apple platforms
  if (APPLE)
    function(export_symbols file)
      target_link_libraries(cxxabi_shared PRIVATE "-Wl,-exported_symbols_list,${file}")
    endfunction()
    function(reexport_symbols file)
      export_symbols("${file}")
      target_link_libraries(cxxabi-reexports INTERFACE "-Wl,-reexported_symbols_list,${file}")
    endfunction()

    export_symbols("${CMAKE_CURRENT_SOURCE_DIR}/../lib/itanium-base.exp")

    if (LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS)
      reexport_symbols("${CMAKE_CURRENT_SOURCE_DIR}/../lib/new-delete.exp")
    endif()

    if (LIBCXXABI_ENABLE_EXCEPTIONS)
      reexport_symbols("${CMAKE_CURRENT_SOURCE_DIR}/../lib/exceptions.exp")

      if ("${CMAKE_OSX_ARCHITECTURES}" MATCHES "^(armv6|armv7|armv7s)$")
        reexport_symbols("${CMAKE_CURRENT_SOURCE_DIR}/../lib/personality-sjlj.exp")
      else()
        reexport_symbols("${CMAKE_CURRENT_SOURCE_DIR}/../lib/personality-v0.exp")
      endif()
    endif()
  endif()
endif()

# Build the static library.
if (LIBCXXABI_ENABLE_STATIC)
  add_library(cxxabi_static STATIC ${LIBCXXABI_SOURCES} ${LIBCXXABI_HEADERS})
  target_link_libraries(cxxabi_static PRIVATE cxx-headers ${LIBCXXABI_STATIC_LIBRARIES} ${LIBCXXABI_LIBRARIES})
  target_link_libraries(cxxabi_static PUBLIC cxxabi-headers)
  if (TARGET pstl::ParallelSTL)
    target_link_libraries(cxxabi_static PUBLIC pstl::ParallelSTL)
  endif()
  set_target_properties(cxxabi_static
    PROPERTIES
      CXX_EXTENSIONS OFF
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED OFF
      COMPILE_FLAGS "${LIBCXXABI_COMPILE_FLAGS}"
      LINK_FLAGS "${LIBCXXABI_LINK_FLAGS}"
      OUTPUT_NAME "c++abi"
    )

  if(LIBCXXABI_HERMETIC_STATIC_LIBRARY)
    append_flags_if_supported(CXXABI_STATIC_LIBRARY_FLAGS -fvisibility=hidden)
    # If the hermetic library doesn't define the operator new/delete functions
    # then its code shouldn't declare them with hidden visibility.  They might
    # actually be provided by a shared library at link time.
    if (LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS)
      append_flags_if_supported(CXXABI_STATIC_LIBRARY_FLAGS -fvisibility-global-new-delete-hidden)
    endif()
    target_compile_options(cxxabi_static PRIVATE ${CXXABI_STATIC_LIBRARY_FLAGS})
    target_compile_definitions(cxxabi_static
      PRIVATE
        _LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS
        _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS)
  endif()

  list(APPEND LIBCXXABI_BUILD_TARGETS "cxxabi_static")
  if (LIBCXXABI_INSTALL_STATIC_LIBRARY)
    list(APPEND LIBCXXABI_INSTALL_TARGETS "cxxabi_static")
  endif()

  if (APPLE)
    set(MERGE_ARCHIVES_LIBTOOL "--use-libtool" "--libtool" "${CMAKE_LIBTOOL}")
  endif()

  # Merge the libc++abi.a and libunwind.a into one.
  if(LIBCXXABI_USE_LLVM_UNWINDER AND LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY)
    add_custom_command(TARGET cxxabi_static POST_BUILD
      COMMAND ${Python3_EXECUTABLE} ${LIBCXXABI_LIBCXX_PATH}/utils/merge_archives.py
      ARGS
        -o "$<TARGET_LINKER_FILE:cxxabi_static>"
        --ar "${CMAKE_AR}"
        ${MERGE_ARCHIVES_LIBTOOL}
        "$<TARGET_LINKER_FILE:cxxabi_static>"
        "$<TARGET_LINKER_FILE:unwind_static>"
      WORKING_DIRECTORY ${LIBCXXABI_BUILD_DIR}
      DEPENDS unwind_static
    )
  endif()
endif()

# Add a meta-target for both libraries.
add_custom_target(cxxabi DEPENDS ${LIBCXXABI_BUILD_TARGETS})

if (LIBCXXABI_INSTALL_LIBRARY)
  install(TARGETS ${LIBCXXABI_INSTALL_TARGETS}
    LIBRARY DESTINATION ${LIBCXXABI_INSTALL_LIBRARY_DIR} COMPONENT cxxabi
    ARCHIVE DESTINATION ${LIBCXXABI_INSTALL_LIBRARY_DIR} COMPONENT cxxabi
    RUNTIME DESTINATION ${LIBCXXABI_INSTALL_RUNTIME_DIR} COMPONENT cxxabi
    )
endif()

if (NOT CMAKE_CONFIGURATION_TYPES AND LIBCXXABI_INSTALL_LIBRARY)
  add_custom_target(install-cxxabi
    DEPENDS cxxabi install-cxxabi-headers
    COMMAND "${CMAKE_COMMAND}"
            -DCMAKE_INSTALL_COMPONENT=cxxabi
            -P "${LIBCXXABI_BINARY_DIR}/cmake_install.cmake")
  add_custom_target(install-cxxabi-stripped
    DEPENDS cxxabi
    COMMAND "${CMAKE_COMMAND}"
            -DCMAKE_INSTALL_COMPONENT=cxxabi
            -DCMAKE_INSTALL_DO_STRIP=1
            -P "${LIBCXXABI_BINARY_DIR}/cmake_install.cmake")
endif()
