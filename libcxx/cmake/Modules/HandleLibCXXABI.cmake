#===============================================================================
# Add an ABI library if appropriate
#===============================================================================

include(GNUInstallDirs)

#
# _setup_abi: Set up the build to use an ABI library
#
# Parameters:
#   abidefines: A list of defines needed to compile libc++ with the ABI library
#   abishared : The shared ABI library to link against.
#   abistatic : The static ABI library to link against.
#   abifiles  : A list of files (which may be relative paths) to copy into the
#               libc++ build tree for the build.  These files will be copied
#               twice: once into include/, so the libc++ build itself can find
#               them, and once into include/c++/v1, so that a clang built into
#               the same build area will find them.
#   abidirs   : A list of relative paths to create under an include directory
#               in the libc++ build directory.
#

macro(setup_abi_lib abidefines abishared abistatic abifiles abidirs)
  list(APPEND LIBCXX_COMPILE_FLAGS ${abidefines})
  set(LIBCXX_CXX_ABI_INCLUDE_PATHS "${LIBCXX_CXX_ABI_INCLUDE_PATHS}"
    CACHE PATH
    "Paths to C++ ABI header directories separated by ';'." FORCE
    )
  set(LIBCXX_CXX_ABI_LIBRARY_PATH "${LIBCXX_CXX_ABI_LIBRARY_PATH}"
    CACHE PATH
    "Paths to C++ ABI library directory"
    )
  set(LIBCXX_CXX_SHARED_ABI_LIBRARY ${abishared})
  set(LIBCXX_CXX_STATIC_ABI_LIBRARY ${abistatic})
  set(LIBCXX_ABILIB_FILES ${abifiles})

  foreach(fpath ${LIBCXX_ABILIB_FILES})
    set(found FALSE)
    foreach(incpath ${LIBCXX_CXX_ABI_INCLUDE_PATHS})
      message(STATUS "Looking for ${fpath} in ${incpath}")
      if (EXISTS "${incpath}/${fpath}")
        set(found TRUE)
        message(STATUS "Looking for ${fpath} in ${incpath} - found")
        get_filename_component(dstdir ${fpath} PATH)
        get_filename_component(ifile ${fpath} NAME)
        set(src ${incpath}/${fpath})

        set(dst ${LIBCXX_BINARY_INCLUDE_DIR}/${dstdir}/${ifile})
        add_custom_command(OUTPUT ${dst}
            DEPENDS ${src}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
            COMMENT "Copying C++ ABI header ${fpath}...")
        list(APPEND abilib_headers "${dst}")

        # TODO: libc++ shouldn't be responsible for copying the libc++abi
        # headers into the right location.
        set(dst "${LIBCXX_GENERATED_INCLUDE_DIR}/${dstdir}/${fpath}")
        add_custom_command(OUTPUT ${dst}
            DEPENDS ${src}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
            COMMENT "Copying C++ ABI header ${fpath}...")
        list(APPEND abilib_headers "${dst}")
      else()
        message(STATUS "Looking for ${fpath} in ${incpath} - not found")
      endif()
    endforeach()
    if (NOT found)
      message(WARNING "Failed to find ${fpath} in ${LIBCXX_CXX_ABI_INCLUDE_PATHS}")
    endif()
  endforeach()

  include_directories("${LIBCXX_BINARY_INCLUDE_DIR}")
  add_custom_target(cxx_abi_headers ALL DEPENDS ${abilib_headers})
  set(LIBCXX_CXX_ABI_HEADER_TARGET "cxx_abi_headers")
endmacro()


# Configure based on the selected ABI library.
if ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libstdc++" OR
    "${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libsupc++")
  set(_LIBSUPCXX_INCLUDE_FILES
    cxxabi.h bits/c++config.h bits/os_defines.h bits/cpu_defines.h
    bits/cxxabi_tweaks.h bits/cxxabi_forced.h
    )
  if ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libstdc++")
    set(_LIBSUPCXX_DEFINES "-DLIBSTDCXX")
    set(_LIBSUPCXX_LIBNAME stdc++)
  else()
    set(_LIBSUPCXX_DEFINES "")
    set(_LIBSUPCXX_LIBNAME supc++)
  endif()
  setup_abi_lib(
    "-D__GLIBCXX__ ${_LIBSUPCXX_DEFINES}"
    "${_LIBSUPCXX_LIBNAME}" "${_LIBSUPCXX_LIBNAME}" "${_LIBSUPCXX_INCLUDE_FILES}" "bits"
    )
elseif ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libcxxabi")
  if(NOT LIBCXX_CXX_ABI_INCLUDE_PATHS)
    set(LIBCXX_CXX_ABI_INCLUDE_PATHS "${LIBCXX_SOURCE_DIR}/../libcxxabi/include")
  endif()

  if(LIBCXX_STANDALONE_BUILD AND NOT (LIBCXX_CXX_ABI_INTREE OR HAVE_LIBCXXABI))
    set(shared c++abi)
    set(static c++abi)
  else()
    set(shared cxxabi_shared)
    set(static cxxabi_static)
  endif()

  setup_abi_lib(
    "-DLIBCXX_BUILDING_LIBCXXABI"
    "${shared}" "${static}" "cxxabi.h;__cxxabi_config.h" "")
elseif ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "system-libcxxabi")
  setup_abi_lib(
    "-DLIBCXX_BUILDING_LIBCXXABI"
    "c++abi" "c++abi" "cxxabi.h;__cxxabi_config.h" "")
elseif ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libcxxrt")
  if(NOT LIBCXX_CXX_ABI_INCLUDE_PATHS)
    set(LIBCXX_CXX_ABI_INCLUDE_PATHS "/usr/include/c++/v1")
  endif()
  # libcxxrt does not provide aligned new and delete operators
  set(LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS ON)
  setup_abi_lib(
    "-DLIBCXXRT"
    "cxxrt" "cxxrt" "cxxabi.h;unwind.h;unwind-arm.h;unwind-itanium.h" ""
    )
elseif ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "vcruntime")
 # Nothing to do
elseif ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "none")
  list(APPEND LIBCXX_COMPILE_FLAGS "-D_LIBCPP_BUILDING_HAS_NO_ABI_LIBRARY")
elseif ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "default")
  # Nothing to do
else()
  message(FATAL_ERROR
    "Unsupported c++ abi: '${LIBCXX_CXX_ABI_LIBNAME}'. \
     Currently libstdc++, libsupc++, libcxxabi, libcxxrt, default and none are
     supported for c++ abi."
    )
endif ()
