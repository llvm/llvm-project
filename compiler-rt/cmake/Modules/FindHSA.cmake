# Find ROCm HSA runtime headers (hsa.h/hsa_ext_amd.h).
#
# This module is used by compiler-rt when SANITIZER_AMDGPU is enabled.
#
# The following variables may be set by the user:
#   HSA_ROOT              ROCm / HSA install prefix (expects include/hsa/hsa.h)
#   ROCM_PATH             Same as typical ROCm install (expects include/hsa/hsa.h)
#   SANITIZER_HSA_INCLUDE_PATH
#                         Legacy hint: directory searched like the original
#                         find_path(HINTS ... PATH_SUFFIXES hsa) entry.
#
# The following cache variables may be set by this module:
#   HSA_INCLUDE_DIR       Directory containing hsa.h (typically .../include/hsa)
#
# This module defines:
#   HSA_FOUND             TRUE if hsa.h was found
#
# Example:
#   find_package(HSA REQUIRED)

include(FindPackageHandleStandardArgs)

set(_hsa_search_paths "")
foreach(_root IN ITEMS "${HSA_ROOT}" "${ROCM_PATH}" "$ENV{ROCM_PATH}")
  if(_root)
    list(APPEND _hsa_search_paths "${_root}/include")
  endif()
endforeach()
if(SANITIZER_HSA_INCLUDE_PATH)
  list(APPEND _hsa_search_paths "${SANITIZER_HSA_INCLUDE_PATH}")
endif()
# Default Legacy ROCm include path
list(APPEND _hsa_search_paths "/opt/rocm/include")

find_path(
  HSA_INCLUDE_DIR
  NAMES hsa.h
  HINTS ${_hsa_search_paths}
  PATH_SUFFIXES hsa
)

# If SANITIZER_HSA_INCLUDE_PATH points at the leaf `hsa` directory, retry
# without PATH_SUFFIXES.
if(NOT HSA_INCLUDE_DIR AND SANITIZER_HSA_INCLUDE_PATH)
  find_path(
    HSA_INCLUDE_DIR
    NAMES hsa.h
    HINTS "${SANITIZER_HSA_INCLUDE_PATH}"
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
  )
endif()

find_package_handle_standard_args(HSA REQUIRED_VARS HSA_INCLUDE_DIR)

mark_as_advanced(HSA_INCLUDE_DIR)
