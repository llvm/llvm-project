# Copy the manifest's libc headers into DEST, preserving their paths, so
# compiler-rt can build the libc-backed builtins against that subset alone.
#
#   cmake -DLIBC_ROOT=<libc> -DDEST=<dir> -DMANIFEST=<file> -P export-libc-shared.cmake

if(NOT LIBC_ROOT OR NOT DEST OR NOT MANIFEST)
  message(FATAL_ERROR "set LIBC_ROOT, DEST and MANIFEST")
endif()

file(STRINGS "${MANIFEST}" _lines)
set(_count 0)
foreach(_rel IN LISTS _lines)
  string(STRIP "${_rel}" _rel)
  if(_rel STREQUAL "" OR _rel MATCHES "^#")
    continue()
  endif()
  if(NOT EXISTS "${LIBC_ROOT}/${_rel}")
    message(FATAL_ERROR "manifest lists a missing file: ${LIBC_ROOT}/${_rel}")
  endif()
  get_filename_component(_dstdir "${DEST}/${_rel}" DIRECTORY)
  file(MAKE_DIRECTORY "${_dstdir}")
  configure_file("${LIBC_ROOT}/${_rel}" "${DEST}/${_rel}" COPYONLY)
  math(EXPR _count "${_count} + 1")
endforeach()
message(STATUS "export-libc-shared: copied ${_count} files into ${DEST}")
