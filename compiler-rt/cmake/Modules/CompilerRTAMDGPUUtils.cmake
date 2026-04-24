# ROCm header discovery for compiler-rt when SANITIZER_AMDGPU is enabled.
#
# Include this module and call:
#   compiler_rt_find_amdgpu_sanitizer_dependencies()
#
# User-settable hints (optional):
#   HSA_ROOT / AMDComgr_ROOT   Install prefix (expects include/hsa/... or include/amd_comgr/...)
#   ROCM_PATH                  Typical ROCm layout; also honors $ENV{ROCM_PATH}
#   SANITIZER_HSA_INCLUDE_PATH Legacy: same search as original find_path(... PATH_SUFFIXES hsa)
#   SANITIZER_COMGR_INCLUDE_PATH
#                              Legacy: same search as original find_path(... PATH_SUFFIXES amd_comgr)
#
# Output (same as the former FindHSA / FindAMDComgr modules):
#   HSA_INCLUDE_DIR, HSA_FOUND
#   AMDComgr_INCLUDE_DIR, AMDComgr_FOUND
#
# This call is REQUIRED-style: missing headers trigger a fatal error from
# find_package_handle_standard_args.

include(FindPackageHandleStandardArgs)

macro(compiler_rt_find_amdgpu_sanitizer_dependencies)
  # --- HSA (hsa.h) ---
  set(_hsa_search_paths "")
  foreach(_root IN ITEMS "${HSA_ROOT}" "${ROCM_PATH}" "$ENV{ROCM_PATH}")
    if(_root)
      list(APPEND _hsa_search_paths "${_root}/include")
    endif()
  endforeach()
  if(SANITIZER_HSA_INCLUDE_PATH)
    list(APPEND _hsa_search_paths "${SANITIZER_HSA_INCLUDE_PATH}")
  endif()
  list(APPEND _hsa_search_paths "/opt/rocm/include")

  find_path(
    HSA_INCLUDE_DIR
    NAMES hsa.h
    HINTS ${_hsa_search_paths}
    PATH_SUFFIXES hsa
  )

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

  # --- AMD COMGR (amd_comgr.h.in / amd_comgr.h) ---
  set(_amdcomgr_search_paths "")
  foreach(_root IN ITEMS "${AMDComgr_ROOT}" "${ROCM_PATH}" "$ENV{ROCM_PATH}")
    if(_root)
      list(APPEND _amdcomgr_search_paths "${_root}/include")
    endif()
  endforeach()
  if(SANITIZER_COMGR_INCLUDE_PATH)
    list(APPEND _amdcomgr_search_paths "${SANITIZER_COMGR_INCLUDE_PATH}")
  endif()
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
endmacro()
