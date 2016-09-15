set(LIBCXX_LIB_CMAKEFILES_DIR "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"  PARENT_SCOPE)

# Get sources
file(GLOB LIBCXX_SOURCES ../src/*.cpp)
if(WIN32)
  file(GLOB LIBCXX_WIN32_SOURCES ../src/support/win32/*.cpp)
  list(APPEND LIBCXX_SOURCES ${LIBCXX_WIN32_SOURCES})
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS")
  file(GLOB LIBCXX_SOLARIS_SOURCES ../src/support/solaris/*.c)
  list(APPEND LIBCXX_SOURCES ${LIBCXX_SOLARIS_SOURCES})
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

#if LIBCXX_CXX_ABI_LIBRARY_PATH is defined we want to add it to the search path.
add_link_flags_if(LIBCXX_CXX_ABI_LIBRARY_PATH "-L${LIBCXX_CXX_ABI_LIBRARY_PATH}")

add_library_flags_if(LIBCXX_COVERAGE_LIBRARY "${LIBCXX_COVERAGE_LIBRARY}")

add_library_flags_if(LIBCXX_ENABLE_STATIC_ABI_LIBRARY "-Wl,--whole-archive" "-Wl,-Bstatic")
add_library_flags("${LIBCXX_CXX_ABI_LIBRARY}")
add_library_flags_if(LIBCXX_ENABLE_STATIC_ABI_LIBRARY "-Wl,-Bdynamic" "-Wl,--no-whole-archive")

if (APPLE AND LLVM_USE_SANITIZER)
  if ("${LLVM_USE_SANITIZER}" STREQUAL "Address")
    set(LIBFILE "libclang_rt.asan_osx_dynamic.dylib")
  elseif("${LLVM_USE_SANITIZER}" STREQUAL "Undefined")
    set(LIBFILE "libclang_rt.ubsan_osx_dynamic.dylib")
  elseif("${LLVM_USE_SANITIZER}" STREQUAL "Thread")
    set(LIBFILE "libclang_rt.tsan_osx_dynamic.dylib")
  else()
    message(WARNING "LLVM_USE_SANITIZER=${LLVM_USE_SANITIZER} is not supported on OS X")
  endif()
  if (LIBFILE)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=lib OUTPUT_VARIABLE LIBDIR RESULT_VARIABLE Result)
    if (NOT ${Result} EQUAL "0")
      message(FATAL "Failed to find library resource directory")
    endif()
    string(STRIP "${LIBDIR}" LIBDIR)
    set(LIBDIR "${LIBDIR}/darwin/")
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

# Generate library list.
add_library_flags_if(LIBCXX_HAS_PTHREAD_LIB pthread)
add_library_flags_if(LIBCXX_HAS_C_LIB c)
add_library_flags_if(LIBCXX_HAS_M_LIB m)
add_library_flags_if(LIBCXX_HAS_RT_LIB rt)
add_library_flags_if(LIBCXX_HAS_GCC_S_LIB gcc_s)
add_library_flags_if(LIBCXX_HAVE_CXX_ATOMICS_WITH_LIB atomic)

# Setup flags.
if (NOT WIN32)
  add_flags_if_supported(-fPIC)
endif()
add_link_flags_if_supported(-nodefaultlibs)

if ( APPLE AND (LIBCXX_CXX_ABI_LIBNAME STREQUAL "libcxxabi" OR
                LIBCXX_CXX_ABI_LIBNAME STREQUAL "none"))
  if (NOT DEFINED LIBCXX_LIBCPPABI_VERSION)
    set(LIBCXX_LIBCPPABI_VERSION "2") # Default value
    execute_process(
      COMMAND xcrun --show-sdk-version
      OUTPUT_VARIABLE sdk_ver
      RESULT_VARIABLE res
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (res EQUAL 0)
      message(STATUS "Found SDK version ${sdk_ver}")
      string(REPLACE "10." "" sdk_ver "${sdk_ver}")
      if (sdk_ver LESS 9)
        set(LIBCXX_LIBCPPABI_VERSION "")
      else()
        set(LIBCXX_LIBCPPABI_VERSION "2")
      endif()
    endif()
  endif()

  if ( CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "10.6" )
    add_definitions(-D__STRICT_ANSI__)
    add_link_flags(
      "-compatibility_version 1"
      "-current_version 1"
      "-install_name /usr/lib/libc++.1.dylib"
      "-Wl,-reexport_library,/usr/lib/libc++abi.dylib"
      "-Wl,-unexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/libc++unexp.exp"
      "/usr/lib/libSystem.B.dylib")
  else()
    if (DEFINED CMAKE_OSX_SYSROOT)
      list(FIND CMAKE_OSX_ARCHITECTURES "armv7" OSX_HAS_ARMV7)
      if (NOT OSX_HAS_ARMV7 EQUAL -1)
        set(OSX_RE_EXPORT_LINE
          "${CMAKE_OSX_SYSROOT}/usr/lib/libc++abi.dylib"
          "-Wl,-reexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/libc++sjlj-abi.exp")
      else()
        set(OSX_RE_EXPORT_LINE
          "-Wl,-reexport_library,${CMAKE_OSX_SYSROOT}/usr/lib/libc++abi.dylib")
      endif()
    else()
      set(OSX_RE_EXPORT_LINE "/usr/lib/libc++abi.dylib -Wl,-reexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/libc++abi${LIBCXX_LIBCPPABI_VERSION}.exp")
    endif()

    add_link_flags(
      "-compatibility_version 1"
      "-install_name /usr/lib/libc++.1.dylib"
      "-Wl,-unexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/libc++unexp.exp"
      "${OSX_RE_EXPORT_LINE}"
      "-Wl,-force_symbols_not_weak_list,${CMAKE_CURRENT_SOURCE_DIR}/notweak.exp"
      "-Wl,-force_symbols_weak_list,${CMAKE_CURRENT_SOURCE_DIR}/weak.exp")
  endif()
endif()

split_list(LIBCXX_COMPILE_FLAGS)
split_list(LIBCXX_LINK_FLAGS)

# Add a object library that contains the compiled source files.
add_library(cxx_objects OBJECT ${exclude_from_all} ${LIBCXX_SOURCES} ${LIBCXX_HEADERS})

set_target_properties(cxx_objects
  PROPERTIES
    COMPILE_FLAGS "${LIBCXX_COMPILE_FLAGS}"
)

set(LIBCXX_TARGETS)

# Build the shared library.
if (LIBCXX_ENABLE_SHARED)
  add_library(cxx_shared SHARED $<TARGET_OBJECTS:cxx_objects>)
  target_link_libraries(cxx_shared ${LIBCXX_LIBRARIES})
  set_target_properties(cxx_shared
    PROPERTIES
      LINK_FLAGS    "${LIBCXX_LINK_FLAGS}"
      OUTPUT_NAME   "c++"
      VERSION       "${LIBCXX_ABI_VERSION}.0"
      SOVERSION     "${LIBCXX_ABI_VERSION}"
  )
  list(APPEND LIBCXX_TARGETS "cxx_shared")
endif()

# Build the static library.
if (LIBCXX_ENABLE_STATIC)
  add_library(cxx_static STATIC $<TARGET_OBJECTS:cxx_objects>)
  target_link_libraries(cxx_static ${LIBCXX_LIBRARIES})
  set_target_properties(cxx_static
    PROPERTIES
      LINK_FLAGS    "${LIBCXX_LINK_FLAGS}"
      OUTPUT_NAME   "c++"
  )
  list(APPEND LIBCXX_TARGETS "cxx_static")
endif()

# Add a meta-target for both libraries.
add_custom_target(cxx DEPENDS ${LIBCXX_TARGETS})

if (DEFINED LIBCXX_CXX_ABI_DEPS)
  add_dependencies(cxx LIBCXX_CXX_ABI_DEPS)
endif()

if (LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY)
  file(GLOB LIBCXX_EXPERIMENTAL_SOURCES ../src/experimental/*.cpp)
  if (LIBCXX_ENABLE_FILESYSTEM)
    file(GLOB LIBCXX_FILESYSTEM_SOURCES ../src/experimental/filesystem/*.cpp)
  endif()
  add_library(cxx_experimental STATIC ${LIBCXX_EXPERIMENTAL_SOURCES} ${LIBCXX_FILESYSTEM_SOURCES})
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

if (LIBCXX_HAS_EXTERNAL_THREAD_API)
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

# Generate a linker script inplace of a libc++.so symlink. Rerun this command
# after cxx builds.
if (LIBCXX_ENABLE_SHARED AND LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
  # Get the name of the ABI library and handle the case where CXXABI_LIBNAME
  # is a target name and not a library. Ex cxxabi_shared.
  set(SCRIPT_ABI_LIBNAME "${LIBCXX_CXX_ABI_LIBRARY}")
  if (SCRIPT_ABI_LIBNAME STREQUAL "cxxabi_shared")
    set(SCRIPT_ABI_LIBNAME "c++abi")
  endif()
  # Generate a linker script inplace of a libc++.so symlink. Rerun this command
  # after cxx builds.
  add_custom_command(TARGET cxx_shared POST_BUILD
    COMMAND
      ${PYTHON_EXECUTABLE} ${LIBCXX_SOURCE_DIR}/utils/gen_link_script/gen_link_script.py
    ARGS
      "$<TARGET_LINKER_FILE:cxx_shared>"
      "${SCRIPT_ABI_LIBNAME}"
    WORKING_DIRECTORY ${LIBCXX_BUILD_DIR}
  )
endif()

if (LIBCXX_INSTALL_LIBRARY)
  if (LIBCXX_INSTALL_EXPERIMENTAL_LIBRARY)
    set(experimental_lib cxx_experimental)
  endif()
  install(TARGETS ${LIBCXX_TARGETS} ${experimental_lib}
    LIBRARY DESTINATION lib${LIBCXX_LIBDIR_SUFFIX} COMPONENT cxx
    ARCHIVE DESTINATION lib${LIBCXX_LIBDIR_SUFFIX} COMPONENT cxx
    )
  # NOTE: This install command must go after the cxx install command otherwise
  # it will not be executed after the library symlinks are installed.
  if (LIBCXX_ENABLE_SHARED AND LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
    # Replace the libc++ filename with $<TARGET_LINKER_FILE:cxx>
    # after we required CMake 3.0.
    install(FILES "${LIBCXX_LIBRARY_DIR}/libc++${CMAKE_SHARED_LIBRARY_SUFFIX}"
      DESTINATION lib${LIBCXX_LIBDIR_SUFFIX}
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
    add_custom_target(install-libcxx DEPENDS install-cxx)
endif()
