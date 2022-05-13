set(LIBCXX_LIB_CMAKEFILES_DIR "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"  PARENT_SCOPE)

# Get sources
set(LIBCXX_SOURCES
  algorithm.cpp
  any.cpp
  assert.cpp
  atomic.cpp
  barrier.cpp
  bind.cpp
  charconv.cpp
  chrono.cpp
  condition_variable.cpp
  condition_variable_destructor.cpp
  exception.cpp
  functional.cpp
  future.cpp
  hash.cpp
  include/apple_availability.h
  include/atomic_support.h
  include/config_elast.h
  include/refstring.h
  include/ryu/common.h
  include/ryu/d2fixed.h
  include/ryu/d2fixed_full_table.h
  include/ryu/d2s.h
  include/ryu/d2s_full_table.h
  include/ryu/d2s_intrinsics.h
  include/ryu/digit_table.h
  include/ryu/f2s.h
  include/ryu/ryu.h
  include/to_chars_floating_point.h
  legacy_pointer_safety.cpp
  memory.cpp
  mutex.cpp
  mutex_destructor.cpp
  new.cpp
  optional.cpp
  random_shuffle.cpp
  ryu/d2fixed.cpp
  ryu/d2s.cpp
  ryu/f2s.cpp
  shared_mutex.cpp
  stdexcept.cpp
  string.cpp
  support/runtime/exception_fallback.ipp
  support/runtime/exception_glibcxx.ipp
  support/runtime/exception_libcxxabi.ipp
  support/runtime/exception_libcxxrt.ipp
  support/runtime/exception_msvc.ipp
  support/runtime/exception_pointer_cxxabi.ipp
  support/runtime/exception_pointer_glibcxx.ipp
  support/runtime/exception_pointer_msvc.ipp
  support/runtime/exception_pointer_unimplemented.ipp
  support/runtime/new_handler_fallback.ipp
  support/runtime/stdexcept_default.ipp
  support/runtime/stdexcept_vcruntime.ipp
  system_error.cpp
  thread.cpp
  typeinfo.cpp
  utility.cpp
  valarray.cpp
  variant.cpp
  vector.cpp
  )

if (LIBCXX_ENABLE_DEBUG_MODE_SUPPORT)
  list(APPEND LIBCXX_SOURCES
    debug.cpp
    legacy_debug_handler.cpp
    )
endif()

if (LIBCXX_ENABLE_RANDOM_DEVICE)
  list(APPEND LIBCXX_SOURCES
    random.cpp
    )
endif()

if (LIBCXX_ENABLE_LOCALIZATION)
  list(APPEND LIBCXX_SOURCES
    include/sso_allocator.h
    ios.cpp
    ios.instantiations.cpp
    iostream.cpp
    locale.cpp
    regex.cpp
    strstream.cpp
    )
endif()

if(LIBCXX_ENABLE_INCOMPLETE_FEATURES)
  list(APPEND LIBCXX_SOURCES
    format.cpp
  )
endif()

if(WIN32)
  list(APPEND LIBCXX_SOURCES
    support/win32/locale_win32.cpp
    support/win32/support.cpp
    )

  if (NOT LIBCXX_HAS_PTHREAD_API)
    list(APPEND LIBCXX_SOURCES
      support/win32/thread_win32.cpp
      )
  endif()
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS")
  list(APPEND LIBCXX_SOURCES
    support/solaris/mbsnrtowcs.inc
    support/solaris/wcsnrtombs.inc
    support/solaris/xlocale.cpp
    )
elseif(ZOS)
  list(APPEND LIBCXX_SOURCES
    support/ibm/mbsnrtowcs.cpp
    support/ibm/wcsnrtombs.cpp
    support/ibm/xlocale_zos.cpp
    )
endif()

if (LIBCXX_ENABLE_FILESYSTEM)
  list(APPEND LIBCXX_SOURCES
    filesystem/filesystem_common.h
    filesystem/operations.cpp
    filesystem/directory_iterator.cpp
    filesystem/posix_compat.h
    )
  # Filesystem uses __int128_t, which requires a definition of __muloi4 when
  # compiled with UBSAN. This definition is not provided by libgcc_s, but is
  # provided by compiler-rt. So we need to disable it to avoid having multiple
  # definitions. See filesystem/int128_builtins.cpp.
  if (NOT LIBCXX_USE_COMPILER_RT)
    list(APPEND LIBCXX_SOURCES
      filesystem/int128_builtins.cpp
      )
  endif()
endif()

# Add all the headers to the project for IDEs.
if (LIBCXX_CONFIGURE_IDE)
  file(GLOB_RECURSE LIBCXX_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../include/*)
  if(WIN32)
    file( GLOB LIBCXX_WIN32_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../include/__support/win32/*.h)
    list(APPEND LIBCXX_HEADERS ${LIBCXX_WIN32_HEADERS})
  endif()
  # Force them all into the headers dir on MSVC, otherwise they end up at
  # project scope because they don't have extensions.
  if (MSVC_IDE)
    source_group("Header Files" FILES ${LIBCXX_HEADERS})
  endif()
endif()

if(NOT LIBCXX_INSTALL_LIBRARY)
  set(exclude_from_all EXCLUDE_FROM_ALL)
endif()

if (LIBCXX_GENERATE_COVERAGE AND NOT LIBCXX_COVERAGE_LIBRARY)
  find_compiler_rt_library(profile LIBCXX_COVERAGE_LIBRARY)
endif()
add_library_flags_if(LIBCXX_COVERAGE_LIBRARY "${LIBCXX_COVERAGE_LIBRARY}")

if (APPLE AND LLVM_USE_SANITIZER)
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
    find_compiler_rt_library(builtins LIBCXX_BUILTINS_LIBRARY)
    get_filename_component(LIBDIR "${LIBCXX_BUILTINS_LIBRARY}" DIRECTORY)
    if (NOT IS_DIRECTORY "${LIBDIR}")
      message(FATAL_ERROR "Cannot find compiler-rt directory on OS X required for LLVM_USE_SANITIZER")
    endif()
    set(LIBCXX_SANITIZER_LIBRARY "${LIBDIR}/${LIBFILE}")
    set(LIBCXX_SANITIZER_LIBRARY "${LIBCXX_SANITIZER_LIBRARY}" PARENT_SCOPE)
    message(STATUS "Manually linking compiler-rt library: ${LIBCXX_SANITIZER_LIBRARY}")
    add_library_flags("${LIBCXX_SANITIZER_LIBRARY}")
    add_link_flags("-Wl,-rpath,${LIBDIR}")
  endif()
endif()

if (LIBCXX_ENABLE_PARALLEL_ALGORITHMS AND NOT TARGET pstl::ParallelSTL)
  message(FATAL_ERROR "Could not find ParallelSTL")
endif()

function(cxx_set_common_defines name)
  if (LIBCXX_ENABLE_PARALLEL_ALGORITHMS)
    target_link_libraries(${name} PUBLIC pstl::ParallelSTL)
  endif()
endfunction()

split_list(LIBCXX_COMPILE_FLAGS)
split_list(LIBCXX_LINK_FLAGS)

# Build the shared library.
if (LIBCXX_ENABLE_SHARED)
  add_library(cxx_shared SHARED ${exclude_from_all} ${LIBCXX_SOURCES} ${LIBCXX_HEADERS})
  target_include_directories(cxx_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(cxx_shared PUBLIC cxx-headers
                                   PRIVATE ${LIBCXX_LIBRARIES})
  set_target_properties(cxx_shared
    PROPERTIES
      COMPILE_FLAGS "${LIBCXX_COMPILE_FLAGS}"
      LINK_FLAGS    "${LIBCXX_LINK_FLAGS}"
      OUTPUT_NAME   "c++"
      VERSION       "${LIBCXX_ABI_VERSION}.0"
      SOVERSION     "${LIBCXX_ABI_VERSION}"
      DEFINE_SYMBOL ""
  )
  cxx_add_common_build_flags(cxx_shared)
  cxx_set_common_defines(cxx_shared)

  # Link against LLVM libunwind
  if (LIBCXXABI_USE_LLVM_UNWINDER)
    if (NOT LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_SHARED_LIBRARY AND (TARGET unwind_shared OR HAVE_LIBUNWIND))
      target_link_libraries(cxx_shared PUBLIC unwind_shared)
    elseif (LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_SHARED_LIBRARY AND (TARGET unwind_static OR HAVE_LIBUNWIND))
      # libunwind is already included in libc++abi
    else()
      target_link_libraries(cxx_shared PUBLIC unwind)
    endif()
  endif()

  # Link against libc++abi
  if (LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY)
    if (APPLE)
      target_link_libraries(cxx_shared PRIVATE "-Wl,-force_load" "$<TARGET_LINKER_FILE:libcxx-abi-static>")
    else()
      target_link_libraries(cxx_shared PRIVATE "-Wl,--whole-archive,-Bstatic" "$<TARGET_LINKER_FILE:libcxx-abi-static>" "-Wl,-Bdynamic,--no-whole-archive")
    endif()
  else()
    target_link_libraries(cxx_shared PUBLIC libcxx-abi-shared)
  endif()

  # Maybe re-export symbols from libc++abi
  # In particular, we don't re-export the symbols if libc++abi is merged statically
  # into libc++ because in that case there's no dylib to re-export from.
  if (APPLE AND LIBCXX_CXX_ABI STREQUAL "libcxxabi"
            AND NOT DEFINED LIBCXX_OSX_REEXPORT_LIBCXXABI_SYMBOLS
            AND NOT LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY)
    set(LIBCXX_OSX_REEXPORT_LIBCXXABI_SYMBOLS ON)
  endif()

  if (LIBCXX_OSX_REEXPORT_LIBCXXABI_SYMBOLS)
    target_link_libraries(cxx_shared PRIVATE
      "-Wl,-unexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/../lib/libc++unexp.exp"
      "-Wl,-reexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/../lib/libc++abi.v${LIBCXX_LIBCPPABI_VERSION}.exp"
      "-Wl,-force_symbols_not_weak_list,${CMAKE_CURRENT_SOURCE_DIR}/../lib/notweak.exp"
      "-Wl,-force_symbols_weak_list,${CMAKE_CURRENT_SOURCE_DIR}/../lib/weak.exp")

    target_link_libraries(cxx_shared PRIVATE $<TARGET_NAME_IF_EXISTS:cxxabi-reexports>)
  endif()

  # Generate a linker script in place of a libc++.so symlink.
  if (LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
      include(DefineLinkerScript)
      define_linker_script(cxx_shared)
  endif()

  list(APPEND LIBCXX_BUILD_TARGETS "cxx_shared")
  if(WIN32 AND NOT MINGW AND NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
    # Since we most likely do not have a mt.exe replacement, disable the
    # manifest bundling.  This allows a normal cmake invocation to pass which
    # will attempt to use the manifest tool to generate the bundled manifest
    set_target_properties(cxx_shared PROPERTIES
                          APPEND_STRING PROPERTY LINK_FLAGS " /MANIFEST:NO")
  endif()
endif()

set(CMAKE_STATIC_LIBRARY_PREFIX "lib")

# Build the static library.
if (LIBCXX_ENABLE_STATIC)
  add_library(cxx_static STATIC ${exclude_from_all} ${LIBCXX_SOURCES} ${LIBCXX_HEADERS})
  target_include_directories(cxx_static PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(cxx_static PUBLIC cxx-headers
                                   PRIVATE ${LIBCXX_LIBRARIES}
                                   PRIVATE libcxx-abi-static)
  set_target_properties(cxx_static
    PROPERTIES
      COMPILE_FLAGS "${LIBCXX_COMPILE_FLAGS}"
      LINK_FLAGS    "${LIBCXX_LINK_FLAGS}"
      OUTPUT_NAME   "c++"
  )
  cxx_add_common_build_flags(cxx_static)
  cxx_set_common_defines(cxx_static)

  if (LIBCXXABI_USE_LLVM_UNWINDER)
    # If we intend to use the just-built unwinder, add a dependency so that it
    # gets built, even if we technically aren't going to link it in at this
    # stage.
    add_dependencies(cxx_static unwind)
  endif()

  if (LIBCXX_HERMETIC_STATIC_LIBRARY)
    # If the hermetic library doesn't define the operator new/delete functions
    # then its code shouldn't declare them with hidden visibility.  They might
    # actually be provided by a shared library at link time.
    if (LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS)
      append_flags_if_supported(CXX_STATIC_LIBRARY_FLAGS -fvisibility-global-new-delete-hidden)
    endif()
    target_compile_options(cxx_static PRIVATE ${CXX_STATIC_LIBRARY_FLAGS})
    target_compile_definitions(cxx_static PRIVATE _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS)
  endif()

  list(APPEND LIBCXX_BUILD_TARGETS "cxx_static")
  # Attempt to merge the libc++.a archive and the ABI library archive into one.
  if (LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY)
    if (APPLE)
      set(MERGE_ARCHIVES_LIBTOOL "--use-libtool" "--libtool" "${CMAKE_LIBTOOL}")
    endif()
    add_custom_command(TARGET cxx_static POST_BUILD
    COMMAND
      ${Python3_EXECUTABLE} ${LIBCXX_SOURCE_DIR}/utils/merge_archives.py
    ARGS
      -o $<TARGET_LINKER_FILE:cxx_static>
      --ar "${CMAKE_AR}"
      ${MERGE_ARCHIVES_LIBTOOL}
      "$<TARGET_LINKER_FILE:cxx_static>"
      "$<TARGET_LINKER_FILE:libcxx-abi-static>"
      ""
    WORKING_DIRECTORY ${LIBCXX_BUILD_DIR}
    DEPENDS ${MERGE_ARCHIVES_ABI_TARGET}
    )
  endif()
endif()

# Add a meta-target for both libraries.
add_custom_target(cxx DEPENDS ${LIBCXX_BUILD_TARGETS})

if (LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY)
  set(LIBCXX_EXPERIMENTAL_SOURCES
    experimental/memory_resource.cpp
    )
  add_library(cxx_experimental STATIC ${LIBCXX_EXPERIMENTAL_SOURCES})
  if (LIBCXX_ENABLE_SHARED)
    target_link_libraries(cxx_experimental PRIVATE cxx_shared)
  else()
    target_link_libraries(cxx_experimental PRIVATE cxx_static)
  endif()

  set_target_properties(cxx_experimental
    PROPERTIES
      COMPILE_FLAGS "${LIBCXX_COMPILE_FLAGS}"
      OUTPUT_NAME   "c++experimental"
  )
  cxx_add_common_build_flags(cxx_experimental)
endif()


if (LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY)
  set(LIBCXX_EXTERNAL_THREADING_SUPPORT_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/../test/support/external_threads.cpp")

  if (LIBCXX_ENABLE_SHARED)
    add_library(cxx_external_threads SHARED ${LIBCXX_EXTERNAL_THREADING_SUPPORT_SOURCES})
  else()
    add_library(cxx_external_threads STATIC ${LIBCXX_EXTERNAL_THREADING_SUPPORT_SOURCES})
  endif()

  set_target_properties(cxx_external_threads
    PROPERTIES
      LINK_FLAGS    "${LIBCXX_LINK_FLAGS}"
      COMPILE_FLAGS "${LIBCXX_COMPILE_FLAGS}"
      OUTPUT_NAME   "c++external_threads"
  )

  target_link_libraries(cxx_external_threads PRIVATE cxx-headers)
endif()

if (LIBCXX_INSTALL_SHARED_LIBRARY)
  install(TARGETS cxx_shared
    ARCHIVE DESTINATION ${LIBCXX_INSTALL_LIBRARY_DIR} COMPONENT cxx
    LIBRARY DESTINATION ${LIBCXX_INSTALL_LIBRARY_DIR} COMPONENT cxx
    RUNTIME DESTINATION ${LIBCXX_INSTALL_RUNTIME_DIR} COMPONENT cxx)
endif()

if (LIBCXX_INSTALL_STATIC_LIBRARY)
  install(TARGETS cxx_static
    ARCHIVE DESTINATION ${LIBCXX_INSTALL_LIBRARY_DIR} COMPONENT cxx
    LIBRARY DESTINATION ${LIBCXX_INSTALL_LIBRARY_DIR} COMPONENT cxx
    RUNTIME DESTINATION ${LIBCXX_INSTALL_RUNTIME_DIR} COMPONENT cxx)
endif()

if(LIBCXX_INSTALL_EXPERIMENTAL_LIBRARY)
  install(TARGETS cxx_experimental
    LIBRARY DESTINATION ${LIBCXX_INSTALL_LIBRARY_DIR} COMPONENT cxx
    ARCHIVE DESTINATION ${LIBCXX_INSTALL_LIBRARY_DIR} COMPONENT cxx
    RUNTIME DESTINATION ${LIBCXX_INSTALL_RUNTIME_DIR} COMPONENT cxx)
endif()

# NOTE: This install command must go after the cxx install command otherwise
# it will not be executed after the library symlinks are installed.
if (LIBCXX_ENABLE_SHARED AND LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
  install(FILES "$<TARGET_LINKER_FILE:cxx_shared>"
    DESTINATION ${LIBCXX_INSTALL_LIBRARY_DIR}
    COMPONENT libcxx)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES)
    if(LIBCXX_INSTALL_LIBRARY)
      set(lib_install_target cxx)
    endif()
    if (LIBCXX_INSTALL_EXPERIMENTAL_LIBRARY)
      set(experimental_lib_install_target cxx_experimental)
    endif()
    if(LIBCXX_INSTALL_HEADERS)
      set(header_install_target install-cxx-headers)
    endif()
    if (LIBCXX_ENABLE_PARALLEL_ALGORITHMS)
      set(pstl_install_target install-pstl)
    endif()
    add_custom_target(install-cxx
                      DEPENDS ${lib_install_target}
                              ${experimental_lib_install_target}
                              ${header_install_target}
                              ${pstl_install_target}
                      COMMAND "${CMAKE_COMMAND}"
                      -DCMAKE_INSTALL_COMPONENT=cxx
                      -P "${LIBCXX_BINARY_DIR}/cmake_install.cmake")
    add_custom_target(install-cxx-stripped
                      DEPENDS ${lib_install_target}
                              ${experimental_lib_install_target}
                              ${header_install_target}
                              ${pstl_install_target}
                      COMMAND "${CMAKE_COMMAND}"
                      -DCMAKE_INSTALL_COMPONENT=cxx
                      -DCMAKE_INSTALL_DO_STRIP=1
                      -P "${LIBCXX_BINARY_DIR}/cmake_install.cmake")
endif()
