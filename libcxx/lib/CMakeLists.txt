set(LIBCXX_LIB_CMAKEFILES_DIR "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"  PARENT_SCOPE)

# Get sources
# FIXME: Don't use glob here
file(GLOB LIBCXX_SOURCES ../src/*.cpp)
if(WIN32)
  file(GLOB LIBCXX_WIN32_SOURCES ../src/support/win32/*.cpp)
  list(APPEND LIBCXX_SOURCES ${LIBCXX_WIN32_SOURCES})
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS")
  file(GLOB LIBCXX_SOLARIS_SOURCES ../src/support/solaris/*.cpp)
  list(APPEND LIBCXX_SOURCES ${LIBCXX_SOLARIS_SOURCES})
endif()

if (LIBCXX_ENABLE_FILESYSTEM)
  list(APPEND LIBCXX_SOURCES ../src/filesystem/operations.cpp
                             ../src/filesystem/directory_iterator.cpp)
  # Filesystem uses __int128_t, which requires a definition of __muloi4 when
  # compiled with UBSAN. This definition is not provided by libgcc_s, but is
  # provided by compiler-rt. So we need to disable it to avoid having multiple
  # definitions. See filesystem/int128_builtins.cpp.
  if (NOT LIBCXX_USE_COMPILER_RT)
    list(APPEND LIBCXX_SOURCES ../src/filesystem/int128_builtins.cpp)
  endif()
endif()

# Add all the headers to the project for IDEs.
if (LIBCXX_CONFIGURE_IDE)
  file(GLOB_RECURSE LIBCXX_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../include/*)
  if(WIN32)
    file( GLOB LIBCXX_WIN32_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../include/support/win32/*.h)
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

# If LIBCXX_CXX_ABI_LIBRARY_PATH is defined we want to add it to the search path.
add_link_flags_if(LIBCXX_CXX_ABI_LIBRARY_PATH
                  "${CMAKE_LIBRARY_PATH_FLAG}${LIBCXX_CXX_ABI_LIBRARY_PATH}")


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
    find_compiler_rt_dir(LIBDIR)
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

# Generate private library list.
add_library_flags_if(LIBCXX_HAS_PTHREAD_LIB pthread)
add_library_flags_if(LIBCXX_HAS_C_LIB c)
add_library_flags_if(LIBCXX_HAS_M_LIB m)
add_library_flags_if(LIBCXX_HAS_RT_LIB rt)
if (LIBCXX_USE_COMPILER_RT)
  find_compiler_rt_library(builtins LIBCXX_BUILTINS_LIBRARY)
  add_library_flags_if(LIBCXX_BUILTINS_LIBRARY "${LIBCXX_BUILTINS_LIBRARY}")
else()
  add_library_flags_if(LIBCXX_HAS_GCC_S_LIB gcc_s)
endif()
add_library_flags_if(LIBCXX_HAVE_CXX_ATOMICS_WITH_LIB atomic)
add_library_flags_if(MINGW "${MINGW_LIBRARIES}")

# Add the unwinder library.
if (LIBCXXABI_USE_LLVM_UNWINDER)
  if (NOT LIBCXXABI_ENABLE_STATIC_UNWINDER AND (TARGET unwind_shared OR HAVE_LIBUNWIND))
    add_interface_library(unwind_shared)
  elseif (LIBCXXABI_ENABLE_STATIC_UNWINDER AND (TARGET unwind_static OR HAVE_LIBUNWIND))
    add_interface_library(unwind_static)
  else()
    add_interface_library(unwind)
  endif()
endif()

# Setup flags.
add_link_flags_if_supported(-nodefaultlibs)

if (LIBCXX_TARGETING_MSVC)
  if (LIBCXX_DEBUG_BUILD)
    set(LIB_SUFFIX "d")
  else()
    set(LIB_SUFFIX "")
  endif()
  add_compile_flags(/Zl)
  add_link_flags(/nodefaultlib)

  add_library_flags(ucrt${LIB_SUFFIX}) # Universal C runtime
  add_library_flags(vcruntime${LIB_SUFFIX}) # C++ runtime
  add_library_flags(msvcrt${LIB_SUFFIX}) # C runtime startup files
  add_library_flags(msvcprt${LIB_SUFFIX}) # C++ standard library. Required for exception_ptr internals.
  # Required for standards-complaint wide character formatting functions
  # (e.g. `printfw`/`scanfw`)
  add_library_flags(iso_stdio_wide_specifiers)
endif()

function(cxx_set_common_defines name)
  if(LIBCXX_CXX_ABI_HEADER_TARGET)
    add_dependencies(${name} ${LIBCXX_CXX_ABI_HEADER_TARGET})
  endif()
  if(WIN32 AND NOT MINGW)
    target_compile_definitions(${name}
                               PRIVATE
                                 # Ignore the -MSC_VER mismatch, as we may build
                                 # with a different compatibility version.
                                 _ALLOW_MSC_VER_MISMATCH
                                 # Don't check the msvcprt iterator debug levels
                                 # as we will define the iterator types; libc++
                                 # uses a different macro to identify the debug
                                 # level.
                                 _ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH
                                 # We are building the c++ runtime, don't pull in
                                 # msvcprt.
                                 _CRTBLD
                                 # Don't warn on the use of "deprecated"
                                 # "insecure" functions which are standards
                                 # specified.
                                 _CRT_SECURE_NO_WARNINGS
                                 # Use the ISO conforming behaviour for conversion
                                 # in printf, scanf.
                                 _CRT_STDIO_ISO_WIDE_SPECIFIERS)
  endif()
endfunction()

split_list(LIBCXX_COMPILE_FLAGS)
split_list(LIBCXX_LINK_FLAGS)

# Build the shared library.
if (LIBCXX_ENABLE_SHARED)
  add_library(cxx_shared SHARED ${exclude_from_all} ${LIBCXX_SOURCES} ${LIBCXX_HEADERS})
  if(COMMAND llvm_setup_rpath)
    llvm_setup_rpath(cxx_shared)
  endif()
  target_link_libraries(cxx_shared PRIVATE ${LIBCXX_LIBRARIES})
  set_target_properties(cxx_shared
    PROPERTIES
      COMPILE_FLAGS "${LIBCXX_COMPILE_FLAGS}"
      LINK_FLAGS    "${LIBCXX_LINK_FLAGS}"
      OUTPUT_NAME   "c++"
      VERSION       "${LIBCXX_ABI_VERSION}.0"
      SOVERSION     "${LIBCXX_ABI_VERSION}"
      DEFINE_SYMBOL ""
  )
  cxx_set_common_defines(cxx_shared)

  # Link against libc++abi
  if (LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY)
    if (APPLE)
      target_link_libraries(cxx_shared PRIVATE "-Wl,-force_load" "$<TARGET_LINKER_FILE:${LIBCXX_CXX_STATIC_ABI_LIBRARY}>")
    else()
      target_link_libraries(cxx_shared PRIVATE "-Wl,--whole-archive,-Bstatic" "$<TARGET_LINKER_FILE:${LIBCXX_CXX_STATIC_ABI_LIBRARY}>" "-Wl,-Bdynamic,--no-whole-archive")
    endif()
  else()
    target_link_libraries(cxx_shared PRIVATE "${LIBCXX_CXX_SHARED_ABI_LIBRARY}")
    list(APPEND LIBCXX_INTERFACE_LIBRARIES "${LIBCXX_CXX_STATIC_ABI_LIBRARY}")
  endif()

  # Maybe re-export symbols from libc++abi
  if (APPLE AND (LIBCXX_CXX_ABI_LIBNAME STREQUAL "libcxxabi" OR
                 LIBCXX_CXX_ABI_LIBNAME STREQUAL "default")
            AND NOT DEFINED LIBCXX_OSX_REEXPORT_LIBCXXABI_SYMBOLS)
    set(LIBCXX_OSX_REEXPORT_LIBCXXABI_SYMBOLS ON)
  endif()

  if (LIBCXX_OSX_REEXPORT_LIBCXXABI_SYMBOLS)
    if ("${CMAKE_OSX_ARCHITECTURES}" MATCHES "^(armv6|armv7|armv7s)$")
      set(RE_EXPORT_LIST "${CMAKE_CURRENT_SOURCE_DIR}/libc++sjlj-abi.v${LIBCXX_LIBCPPABI_VERSION}.exp")
    else()
      set(RE_EXPORT_LIST "${CMAKE_CURRENT_SOURCE_DIR}/libc++abi.v${LIBCXX_LIBCPPABI_VERSION}.exp")
    endif()
    target_link_libraries(cxx_shared PRIVATE
      "-compatibility_version 1"
      "-install_name /usr/lib/libc++.1.dylib"
      "-Wl,-unexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/libc++unexp.exp"
      "-Wl,-reexported_symbols_list,${RE_EXPORT_LIST}"
      "-Wl,-force_symbols_not_weak_list,${CMAKE_CURRENT_SOURCE_DIR}/notweak.exp"
      "-Wl,-force_symbols_weak_list,${CMAKE_CURRENT_SOURCE_DIR}/weak.exp")

    if (NOT LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS)
      target_link_libraries(cxx_shared PRIVATE "-Wl,-reexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/libc++abi-new-delete.exp")
    endif()
  endif()

  # Generate a linker script in place of a libc++.so symlink. Rerun this command
  # after cxx builds.
  if (LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
    # Get the name of the ABI library and handle the case where CXXABI_LIBNAME
    # is a target name and not a library. Ex cxxabi_shared.
    set(LIBCXX_INTERFACE_LIBRARY_NAMES)
    foreach(lib ${LIBCXX_INTERFACE_LIBRARIES})
      # FIXME: Handle cxxabi_static and unwind_static.
      if (TARGET ${lib} OR
          (${lib} MATCHES "cxxabi(_static|_shared)?" AND HAVE_LIBCXXABI) OR
          (${lib} MATCHES "unwind(_static|_shared)?" AND HAVE_LIBUNWIND))
        list(APPEND LIBCXX_INTERFACE_LIBRARY_NAMES "$<TARGET_PROPERTY:${lib},OUTPUT_NAME>")
      else()
        list(APPEND LIBCXX_INTERFACE_LIBRARY_NAMES "${lib}")
      endif()
    endforeach()
    add_custom_command(TARGET cxx_shared POST_BUILD
      COMMAND
        ${PYTHON_EXECUTABLE} ${LIBCXX_SOURCE_DIR}/utils/gen_link_script.py
      ARGS
        "$<TARGET_LINKER_FILE:cxx_shared>"
        ${LIBCXX_INTERFACE_LIBRARY_NAMES}
      WORKING_DIRECTORY ${LIBCXX_BUILD_DIR}
    )
  endif()

  list(APPEND LIBCXX_BUILD_TARGETS "cxx_shared")
  if (LIBCXX_INSTALL_SHARED_LIBRARY)
    list(APPEND LIBCXX_INSTALL_TARGETS "cxx_shared")
  endif()
  if(WIN32 AND NOT MINGW AND NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
    # Since we most likely do not have a mt.exe replacement, disable the
    # manifest bundling.  This allows a normal cmake invocation to pass which
    # will attempt to use the manifest tool to generate the bundled manifest
    set_target_properties(cxx_shared PROPERTIES
                          APPEND_STRING PROPERTY LINK_FLAGS " /MANIFEST:NO")
  endif()
endif()

# Build the static library.
if (LIBCXX_ENABLE_STATIC)
  add_library(cxx_static STATIC ${exclude_from_all} ${LIBCXX_SOURCES} ${LIBCXX_HEADERS})
  target_link_libraries(cxx_static PRIVATE ${LIBCXX_LIBRARIES})
  set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
  set_target_properties(cxx_static
    PROPERTIES
      COMPILE_FLAGS "${LIBCXX_COMPILE_FLAGS}"
      LINK_FLAGS    "${LIBCXX_LINK_FLAGS}"
      OUTPUT_NAME   "c++"
  )
  cxx_set_common_defines(cxx_static)

  if (LIBCXX_HERMETIC_STATIC_LIBRARY)
    append_flags_if_supported(CXX_STATIC_LIBRARY_FLAGS -fvisibility=hidden)
    append_flags_if_supported(CXX_STATIC_LIBRARY_FLAGS -fvisibility-global-new-delete-hidden)
    target_compile_options(cxx_static PRIVATE ${CXX_STATIC_LIBRARY_FLAGS})
    target_compile_definitions(cxx_static PRIVATE _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS)
  endif()

  list(APPEND LIBCXX_BUILD_TARGETS "cxx_static")
  if (LIBCXX_INSTALL_STATIC_LIBRARY)
    list(APPEND LIBCXX_INSTALL_TARGETS "cxx_static")
  endif()
  # Attempt to merge the libc++.a archive and the ABI library archive into one.
  if (LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY)
    set(MERGE_ARCHIVES_SEARCH_PATHS "")
    if (LIBCXX_CXX_ABI_LIBRARY_PATH)
      set(MERGE_ARCHIVES_SEARCH_PATHS "-L${LIBCXX_CXX_ABI_LIBRARY_PATH}")
    endif()
    if (TARGET "${LIBCXX_CXX_STATIC_ABI_LIBRARY}" OR HAVE_LIBCXXABI)
      set(MERGE_ARCHIVES_ABI_TARGET "$<TARGET_LINKER_FILE:${LIBCXX_CXX_STATIC_ABI_LIBRARY}>")
    else()
      set(MERGE_ARCHIVES_ABI_TARGET
        "${CMAKE_STATIC_LIBRARY_PREFIX}${LIBCXX_CXX_STATIC_ABI_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()
    add_custom_command(TARGET cxx_static POST_BUILD
    COMMAND
      ${PYTHON_EXECUTABLE} ${LIBCXX_SOURCE_DIR}/utils/merge_archives.py
    ARGS
      -o $<TARGET_LINKER_FILE:cxx_static>
      --ar "${CMAKE_AR}"
      "$<TARGET_LINKER_FILE:cxx_static>"
      "${MERGE_ARCHIVES_ABI_TARGET}"
      "${MERGE_ARCHIVES_SEARCH_PATHS}"
    WORKING_DIRECTORY ${LIBCXX_BUILD_DIR}
    )
  endif()
endif()

# Add a meta-target for both libraries.
add_custom_target(cxx DEPENDS cxx-headers ${LIBCXX_BUILD_TARGETS})

if (LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY)
  file(GLOB LIBCXX_EXPERIMENTAL_SOURCES ../src/experimental/*.cpp)
  add_library(cxx_experimental STATIC ${LIBCXX_EXPERIMENTAL_SOURCES})
  if (LIBCXX_ENABLE_SHARED)
    target_link_libraries(cxx_experimental cxx_shared)
  else()
    target_link_libraries(cxx_experimental cxx_static)
  endif()

  set(experimental_flags "${LIBCXX_COMPILE_FLAGS}")
  check_flag_supported(-std=c++14)
  if (NOT MSVC AND LIBCXX_SUPPORTS_STD_EQ_CXX14_FLAG)
    string(REPLACE "-std=c++11" "-std=c++14" experimental_flags "${LIBCXX_COMPILE_FLAGS}")
  endif()
  set_target_properties(cxx_experimental
    PROPERTIES
      COMPILE_FLAGS "${experimental_flags}"
      OUTPUT_NAME   "c++experimental"
  )
endif()

if (LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY)
  file(GLOB LIBCXX_EXTERNAL_THREADING_SUPPORT_SOURCES ../test/support/external_threads.cpp)

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
endif()

if (LIBCXX_INSTALL_LIBRARY)
  if (LIBCXX_INSTALL_EXPERIMENTAL_LIBRARY)
    set(experimental_lib cxx_experimental)
  endif()
  install(TARGETS ${LIBCXX_INSTALL_TARGETS} ${experimental_lib}
    LIBRARY DESTINATION ${LIBCXX_INSTALL_PREFIX}lib${LIBCXX_LIBDIR_SUFFIX} COMPONENT cxx
    ARCHIVE DESTINATION ${LIBCXX_INSTALL_PREFIX}lib${LIBCXX_LIBDIR_SUFFIX} COMPONENT cxx
    )
  # NOTE: This install command must go after the cxx install command otherwise
  # it will not be executed after the library symlinks are installed.
  if (LIBCXX_ENABLE_SHARED AND LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
    # Replace the libc++ filename with $<TARGET_LINKER_FILE:cxx>
    # after we required CMake 3.0.
    install(FILES "${LIBCXX_LIBRARY_DIR}/libc++${CMAKE_SHARED_LIBRARY_SUFFIX}"
      DESTINATION ${LIBCXX_INSTALL_PREFIX}lib${LIBCXX_LIBDIR_SUFFIX}
      COMPONENT libcxx)
  endif()
endif()

if (NOT CMAKE_CONFIGURATION_TYPES AND (LIBCXX_INSTALL_LIBRARY OR
                                       LIBCXX_INSTALL_HEADERS))
    if(LIBCXX_INSTALL_LIBRARY)
      set(lib_install_target cxx)
    endif()
    if (LIBCXX_INSTALL_EXPERIMENTAL_LIBRARY)
      set(experimental_lib_install_target cxx_experimental)
    endif()
    if(LIBCXX_INSTALL_HEADERS)
      set(header_install_target install-cxx-headers)
    endif()
    add_custom_target(install-cxx
                      DEPENDS ${lib_install_target}
                              ${experimental_lib_install_target}
                              ${header_install_target}
                      COMMAND "${CMAKE_COMMAND}"
                      -DCMAKE_INSTALL_COMPONENT=cxx
                      -P "${LIBCXX_BINARY_DIR}/cmake_install.cmake")
    add_custom_target(install-cxx-stripped
                      DEPENDS ${lib_install_target}
                              ${experimental_lib_install_target}
                              ${header_install_target}
                      COMMAND "${CMAKE_COMMAND}"
                      -DCMAKE_INSTALL_COMPONENT=cxx
                      -DCMAKE_INSTALL_DO_STRIP=1
                      -P "${LIBCXX_BINARY_DIR}/cmake_install.cmake")
    add_custom_target(install-libcxx DEPENDS install-cxx)
endif()

# TODO: This is needed by cxx-benchmarks but this variable isn't
# available outside of the scope of this file so we need to export
# it. This is not necessarily the cleanest solution.
set(LIBCXX_LIBRARIES ${LIBCXX_LIBRARIES} PARENT_SCOPE)
