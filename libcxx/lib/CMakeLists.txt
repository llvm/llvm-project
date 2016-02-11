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

if (LIBCXX_ENABLE_SHARED)
  add_library(cxx SHARED ${exclude_from_all} ${LIBCXX_SOURCES} ${LIBCXX_HEADERS})
else()
  add_library(cxx STATIC ${exclude_from_all} ${LIBCXX_SOURCES} ${LIBCXX_HEADERS})
endif()

if (DEFINED LIBCXX_CXX_ABI_DEPS)
  add_dependencies(cxx LIBCXX_CXX_ABI_DEPS)
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
add_library_flags_if(LIBCXX_HAS_ATOMIC_LIB atomic)

# Setup flags.
add_flags_if_supported(-fPIC)
add_link_flags_if_supported(-nodefaultlibs)

if ( APPLE AND (LIBCXX_CXX_ABI_LIBNAME STREQUAL "libcxxabi" OR
                LIBCXX_CXX_ABI_LIBNAME STREQUAL "none"))
  if (NOT DEFINED LIBCXX_LIBCPPABI_VERSION)
    set(LIBCXX_LIBCPPABI_VERSION "2")
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
    if ( ${CMAKE_OSX_SYSROOT} )
      list(FIND ${CMAKE_OSX_ARCHITECTURES} "armv7" OSX_HAS_ARMV7)
      if (OSX_HAS_ARMV7)
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

target_link_libraries(cxx ${LIBCXX_LIBRARIES})
split_list(LIBCXX_COMPILE_FLAGS)
split_list(LIBCXX_LINK_FLAGS)

set_target_properties(cxx
  PROPERTIES
    COMPILE_FLAGS "${LIBCXX_COMPILE_FLAGS}"
    LINK_FLAGS    "${LIBCXX_LINK_FLAGS}"
    OUTPUT_NAME   "c++"
    VERSION       "${LIBCXX_ABI_VERSION}.0"
    SOVERSION     "${LIBCXX_ABI_VERSION}"
  )

# Generate a linker script inplace of a libc++.so symlink. Rerun this command
# after cxx builds.
if (LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
  # Get the name of the ABI library and handle the case where CXXABI_LIBNAME
  # is a target name and not a library. Ex cxxabi_shared.
  set(SCRIPT_ABI_LIBNAME "${LIBCXX_CXX_ABI_LIBRARY}")
  if (SCRIPT_ABI_LIBNAME STREQUAL "cxxabi_shared")
    set(SCRIPT_ABI_LIBNAME "c++abi")
  endif()
  # Generate a linker script inplace of a libc++.so symlink. Rerun this command
  # after cxx builds.
  add_custom_command(TARGET cxx POST_BUILD
    COMMAND
      ${PYTHON_EXECUTABLE} ${LIBCXX_SOURCE_DIR}/utils/gen_link_script/gen_link_script.py
    ARGS
      "$<TARGET_LINKER_FILE:cxx>"
      "${SCRIPT_ABI_LIBNAME}"
    WORKING_DIRECTORY ${LIBCXX_BUILD_DIR}
  )
endif()

if (LIBCXX_INSTALL_LIBRARY)
  install(TARGETS cxx
    LIBRARY DESTINATION lib${LIBCXX_LIBDIR_SUFFIX} COMPONENT libcxx
    ARCHIVE DESTINATION lib${LIBCXX_LIBDIR_SUFFIX} COMPONENT libcxx
    )
  # NOTE: This install command must go after the cxx install command otherwise
  # it will not be executed after the library symlinks are installed.
  if (LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
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
      set(deps DEPENDS cxx)
    endif()
    add_custom_target(install-libcxx
                      ${deps}
                      COMMAND "${CMAKE_COMMAND}"
                      -DCMAKE_INSTALL_COMPONENT=libcxx
                      -P "${LIBCXX_BINARY_DIR}/cmake_install.cmake")
endif()
