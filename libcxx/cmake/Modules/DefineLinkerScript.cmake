# This function defines a linker script in place of the symlink traditionally
# created for shared libraries.
#
# More specifically, this function goes through the PUBLIC and INTERFACE
# library dependencies of <target> and gathers them into a linker script,
# such that those libraries are linked against when the shared library for
# <target> is linked against.
#
# Arguments:
#   <target>: A target representing a shared library. A linker script will be
#             created in place of that target's TARGET_LINKER_FILE, which is
#             the symlink pointing to the actual shared library (usually
#             libFoo.so pointing to libFoo.so.1, which itself points to
#             libFoo.so.1.0).

function(define_linker_script target)
  if (NOT TARGET "${target}")
    message(FATAL_ERROR "The provided target '${target}' is not actually a target.")
  endif()

  get_target_property(target_type "${target}" TYPE)
  if (NOT "${target_type}" STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR "The provided target '${target}' is not a shared library (its type is '${target_type}').")
  endif()

  set(symlink "$<TARGET_LINKER_FILE:${target}>")
  set(soname "$<TARGET_SONAME_FILE_NAME:${target}>")

  get_target_property(interface_libs "${target}" INTERFACE_LINK_LIBRARIES)

  set(link_libraries)
  if (interface_libs)
    foreach(lib IN LISTS interface_libs)
      if ("${lib}" STREQUAL "cxx-headers")
        continue()
      endif()
      if (TARGET "${lib}" OR
          (${lib} MATCHES "cxxabi(_static|_shared)?" AND HAVE_LIBCXXABI) OR
          (${lib} MATCHES "unwind(_static|_shared)?" AND HAVE_LIBUNWIND))
        list(APPEND link_libraries "${CMAKE_LINK_LIBRARY_FLAG}$<TARGET_PROPERTY:${lib},OUTPUT_NAME>")
      else()
        list(APPEND link_libraries "${CMAKE_LINK_LIBRARY_FLAG}${lib}")
      endif()
    endforeach()
  endif()
  string(REPLACE ";" " " link_libraries "${link_libraries}")

  set(linker_script "INPUT(${soname} ${link_libraries})")
  add_custom_command(TARGET "${target}" POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E remove "${symlink}"
    COMMAND "${CMAKE_COMMAND}" -E echo "${linker_script}" > "${symlink}"
    COMMENT "Generating linker script: '${linker_script}' as file ${symlink}"
    VERBATIM
  )
endfunction()
