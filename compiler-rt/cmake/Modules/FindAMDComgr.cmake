# Find AMD COMGR headers (amd_comgr.h or amd_comgr.h.in).
#
# This module is used by compiler-rt when SANITIZER_AMDGPU is enabled.
#
# The following variables may be set by the user:
#   AMDComgr_ROOT         ROCm / amd_comgr install prefix
#   ROCM_PATH             Typical ROCm install (expects include/amd_comgr/...)
#   SANITIZER_COMGR_INCLUDE_PATH
#                         Legacy hint: base directory searched with
#                         PATH_SUFFIXES amd_comgr (same as original CMake).
#
# The following cache variables may be set by this module:
#   AMDComgr_INCLUDE_DIR  Directory containing amd_comgr.h(.in)
#
# This module defines:
#   AMDComgr_FOUND        TRUE if a supported amd_comgr header was found
#
# Example:
#   find_package(AMDComgr REQUIRED)

include(FindPackageHandleStandardArgs)

set(_amdcomgr_search_paths "")
foreach(_root IN ITEMS "${AMDComgr_ROOT}" "${ROCM_PATH}" "$ENV{ROCM_PATH}")
  if(_root)
    list(APPEND _amdcomgr_search_paths "${_root}/include")
  endif()
endforeach()
if(SANITIZER_COMGR_INCLUDE_PATH)
  list(APPEND _amdcomgr_search_paths "${SANITIZER_COMGR_INCLUDE_PATH}")
endif()
# Default Legacy ROCm include path
list(APPEND _amdcomgr_search_paths "/opt/rocm/include")

find_path(
  AMDComgr_INCLUDE_DIR
  NAMES amd_comgr.h.in
  HINTS ${_amdcomgr_search_paths}
  PATH_SUFFIXES amd_comgr
)

if(NOT AMDComgr_INCLUDE_DIR)
  find_path(
    AMDComgr_INCLUDE_DIR
    NAMES amd_comgr.h
    HINTS ${_amdcomgr_search_paths}
    PATH_SUFFIXES amd_comgr
  )
endif()

# If SANITIZER_COMGR_INCLUDE_PATH points at the leaf `amd_comgr` directory,
# retry without PATH_SUFFIXES.
if(NOT AMDComgr_INCLUDE_DIR AND SANITIZER_COMGR_INCLUDE_PATH)
  find_path(
    AMDComgr_INCLUDE_DIR
    NAMES amd_comgr.h.in amd_comgr.h
    HINTS "${SANITIZER_COMGR_INCLUDE_PATH}"
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
  )
endif()

find_package_handle_standard_args(AMDComgr REQUIRED_VARS AMDComgr_INCLUDE_DIR)

mark_as_advanced(AMDComgr_INCLUDE_DIR)
