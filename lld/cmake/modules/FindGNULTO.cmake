# Attempts to discover GCC plugin API.
#
# Example usage:
#
# find_package(GNULTO)
#
# If successful, the following variables will be defined:
# HAVE_GNULTO_H
#
# HAVE_LDPT_REGISTER_CLAIM_FILE_HOOK_V2_SRC is defined depending on the
# features available in plugin-api.h.

find_path(GNULTO_INCLUDE_DIRS plugin-api.h PATHS ${GNULTO_INCLUDE_DIR})
if( EXISTS "${GNULTO_INCLUDE_DIRS}/plugin-api.h" )
  set(HAVE_GNULTO_H 1 CACHE INTERNAL "")

  include(CMakePushCheckState)
  cmake_push_check_state()
  set(HAVE_LDPT_REGISTER_CLAIM_FILE_HOOK_V2_SRC [=[
    #include <stdint.h>
    #include <stddef.h>
    #include <plugin-api.h>
    void test(struct ld_plugin_tv *tv) {
      tv->tv_register_claim_file_v2 = NULL;
    }
    ]=])
  if(DEFINED CMAKE_C_COMPILER)
    include(CheckCSourceCompiles)
    check_c_source_compiles("${HAVE_LDPT_REGISTER_CLAIM_FILE_HOOK_V2_SRC}" HAVE_LDPT_REGISTER_CLAIM_FILE_HOOK_V2)
  else()
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("${HAVE_LDPT_REGISTER_CLAIM_FILE_HOOK_V2_SRC}" HAVE_LDPT_REGISTER_CLAIM_FILE_HOOK_V2)
  endif()
  cmake_pop_check_state()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GNULTO DEFAULT_MSG)
