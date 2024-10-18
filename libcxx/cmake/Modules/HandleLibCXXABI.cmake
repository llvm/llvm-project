#===============================================================================
# Define targets for linking against the selected ABI library
#
# After including this file, the following targets are defined:
# - libcxx-abi-shared: A target representing the selected ABI library for linking into
#                      the libc++ shared library.
# - libcxx-abi-static: A target representing the selected ABI library for linking into
#                      the libc++ static library.
#===============================================================================

include(GNUInstallDirs)

# This function copies the provided headers to a private directory and adds that
# path to the given INTERFACE target. That target can then be linked against to
# get access to those headers (and only those).
#
# The problem this solves is that when building against a system-provided ABI library,
# the ABI headers might live side-by-side with an actual C++ Standard Library
# installation. For that reason, we can't just add `-I <path-to-ABI-headers>`,
# since we would end up also adding the system-provided C++ Standard Library to
# the search path. Instead, what we do is copy just the ABI library headers to
# a private directory and add just that path when we build libc++.
function(import_private_headers target include_dirs headers)
  if (NOT ${include_dirs})
    message(FATAL_ERROR "Missing include paths for the selected ABI library!")
  endif()
  foreach(header ${headers})
    set(found FALSE)
    foreach(incpath ${include_dirs})
      if (EXISTS "${incpath}/${header}")
        set(found TRUE)
        message(STATUS "Looking for ${header} in ${incpath} - found")
        get_filename_component(dstdir ${header} PATH)
        get_filename_component(header_file ${header} NAME)
        set(src ${incpath}/${header})
        set(dst "${LIBCXX_BINARY_DIR}/private-abi-headers/${target}/${dstdir}/${header_file}")

        add_custom_command(OUTPUT ${dst}
            DEPENDS ${src}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBCXX_BINARY_DIR}/private-abi-headers/${target}/${dstdir}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
            COMMENT "Copying C++ ABI header ${header}")
        list(APPEND abilib_headers "${dst}")
      else()
        message(STATUS "Looking for ${header} in ${incpath} - not found")
      endif()
    endforeach()
    if (NOT found)
      message(WARNING "Failed to find ${header} in ${include_dirs}")
    endif()
  endforeach()

  # Work around https://gitlab.kitware.com/cmake/cmake/-/issues/18399
  add_library(${target}-generate-private-headers OBJECT ${abilib_headers})
  set_target_properties(${target}-generate-private-headers PROPERTIES LINKER_LANGUAGE CXX)

  target_link_libraries(${target} INTERFACE ${target}-generate-private-headers)
  target_include_directories(${target} INTERFACE "${LIBCXX_BINARY_DIR}/private-abi-headers/${target}")
endfunction()

# This function creates an imported static library named <target>.
# It imports a library named <name> searched at the given <path>.
function(import_static_library target path name)
  add_library(${target} STATIC IMPORTED GLOBAL)
  find_library(file
    NAMES "${CMAKE_STATIC_LIBRARY_PREFIX}${name}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    PATHS "${path}"
    NO_CACHE)
  set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${file}")
endfunction()

# This function creates a library target for linking against an external ABI library.
#
# <target>: The name of the target to create
# <name>: The name of the library file to search for (e.g. c++abi, stdc++, cxxrt, supc++)
# <type>: Whether to set up a static or a shared library (e.g. SHARED or STATIC)
# <merged>: Whether to include the ABI library's object files directly into libc++. Only makes sense for a static ABI library.
function(import_external_abi_library target name type merged)
  if (${merged} AND "${type}" STREQUAL "SHARED")
    message(FATAL_ERROR "Can't import an external ABI library for merging when requesting a shared ABI library.")
  endif()

  if ("${type}" STREQUAL "SHARED")
    add_library(${target} INTERFACE IMPORTED GLOBAL)
    set_target_properties(${target} PROPERTIES IMPORTED_LIBNAME "${name}")
  elseif ("${type}" STREQUAL "STATIC")
    if (${merged})
      import_static_library(${target}-impl "${LIBCXX_CXX_ABI_LIBRARY_PATH}" ${name})
      add_library(${target} INTERFACE)
      if (APPLE)
        target_link_options(${target} INTERFACE
          "-Wl,-force_load" "$<TARGET_PROPERTY:${target}-impl,IMPORTED_LOCATION>")
      else()
        target_link_options(${target} INTERFACE
          "-Wl,--whole-archive" "-Wl,-Bstatic"
          "$<TARGET_PROPERTY:${target}-impl,IMPORTED_LOCATION>"
          "-Wl,-Bdynamic" "-Wl,--no-whole-archive")
      endif()
    else()
      import_static_library(${target} "${LIBCXX_CXX_ABI_LIBRARY_PATH}" ${name})
    endif()
  endif()
endfunction()

# This function parses an ABI library choice (including optional consumption specifiers)
# and generates a target representing the ABI library to link against.
#
# When a merged ABI library is requested, we only look for a static ABI library because
# doing otherwise makes no sense. Otherwise, we search for the same type of ABI library
# as we're linking into, i.e. we search for a shared ABI library when linking the shared
# libc++ library, and a static ABI library otherwise.
#
# <abi_target>: The name of the target to create
# <linked_into>: Whether this ABI library is linked into a STATIC or SHARED libc++
# <input>: Input to parse as an ABI library choice with an optional consumption specifier.
function(setup_abi_library abi_target linked_into input)
  if ("${input}" MATCHES "^merged-(.+)$")
    set(merged TRUE)
    set(search_type "STATIC")
  elseif ("${input}" MATCHES "^static-(.+)$")
    set(merged FALSE)
    set(search_type "STATIC")
  elseif ("${input}" MATCHES "^shared-(.+)$")
    set(merged FALSE)
    set(search_type "SHARED")
  else()
    set(merged FALSE)
    set(search_type "${linked_into}")
  endif()

  # Strip the consumption specifier from the name, which leaves the name of the standard library.
  string(REGEX REPLACE "^(merged-|static-|shared-)" "" stdlib "${input}")

  # Link against a system-provided libstdc++
  if ("${stdlib}" STREQUAL "libstdc++")
    import_external_abi_library(${abi_target} stdc++ ${search_type} ${merged})
    import_private_headers(${abi_target} "${LIBCXX_CXX_ABI_INCLUDE_PATHS}"
      "cxxabi.h;bits/c++config.h;bits/os_defines.h;bits/cpu_defines.h;bits/cxxabi_tweaks.h;bits/cxxabi_forced.h")
    target_compile_definitions(${abi_target} INTERFACE "-DLIBSTDCXX" "-D__GLIBCXX__")

  # Link against a system-provided libsupc++
  elseif ("${stdlib}" STREQUAL "libsupc++")
    import_external_abi_library(${abi_target} supc++ ${search_type} ${merged})
    import_private_headers(${abi_target} "${LIBCXX_CXX_ABI_INCLUDE_PATHS}"
      "cxxabi.h;bits/c++config.h;bits/os_defines.h;bits/cpu_defines.h;bits/cxxabi_tweaks.h;bits/cxxabi_forced.h")
    target_compile_definitions(${abi_target} INTERFACE "-D__GLIBCXX__")

  # Link against a system-provided libcxxrt
  elseif ("${stdlib}" STREQUAL "libcxxrt")
    import_external_abi_library(${abi_target} cxxrt ${search_type} ${merged})
    import_private_headers(${abi_target} "${LIBCXX_CXX_ABI_INCLUDE_PATHS}"
      "cxxabi.h;unwind.h;unwind-arm.h;unwind-itanium.h")
    target_compile_definitions(${abi_target} INTERFACE "-DLIBCXXRT")

  # Link against a system-provided vcruntime
  elseif ("${stdlib}" STREQUAL "vcruntime")
    # FIXME: Figure out how to configure the ABI library on Windows.
    add_library(${abi_target} INTERFACE)

  # Link against a system-provided libc++abi
  elseif ("${stdlib}" STREQUAL "system-libcxxabi")
    import_external_abi_library(${abi_target} c++abi ${search_type} ${merged})
    import_private_headers(${abi_target} "${LIBCXX_CXX_ABI_INCLUDE_PATHS}"
      "cxxabi.h;__cxxabi_config.h")
    target_compile_definitions(${abi_target} INTERFACE "-DLIBCXX_BUILDING_LIBCXXABI")

  # Link against the in-tree libc++abi.
  elseif ("${stdlib}" STREQUAL "libcxxabi")
    if (${merged})
      get_target_property(_outdir cxxabi_static ARCHIVE_OUTPUT_DIRECTORY)
      get_target_property(_outname cxxabi_static OUTPUT_NAME)
      set(LIBCXX_CXX_ABI_LIBRARY_PATH "${_outdir}")
      import_external_abi_library(${abi_target} "${_outname}" STATIC TRUE)
    else()
      string(TOLOWER "${search_type}" type)
      add_library(${abi_target} INTERFACE)
      target_link_libraries(${abi_target} INTERFACE cxxabi_${type})

      # Populate the OUTPUT_NAME property of the target because that is used when
      # generating a linker script.
      get_target_property(_outname cxxabi_${type} OUTPUT_NAME)
      set_target_properties(${abi_target} PROPERTIES "OUTPUT_NAME" "${_outname}")
    endif()

    # When using the in-tree libc++abi as an ABI library, libc++ re-exports the
    # libc++abi symbols (on platforms where it can) because libc++abi is only an
    # implementation detail of libc++.
    target_link_libraries(${abi_target} INTERFACE cxxabi-reexports)

    target_compile_definitions(${abi_target} INTERFACE "LIBCXX_BUILDING_LIBCXXABI")

  # Don't link against any ABI library
  elseif ("${stdlib}" STREQUAL "none")
    add_library(${abi_target} INTERFACE)
    target_compile_definitions(${abi_target} INTERFACE "-D_LIBCPP_BUILDING_HAS_NO_ABI_LIBRARY")

  else()
    message(FATAL_ERROR "Unsupported C++ ABI library selection: Got ABI selection '${input}', which parsed into standard library '${stdlib}'.")
  endif()
endfunction()

setup_abi_library(libcxx-abi-shared SHARED "${LIBCXX_ABILIB_FOR_SHARED}")
setup_abi_library(libcxx-abi-static STATIC "${LIBCXX_ABILIB_FOR_STATIC}")
