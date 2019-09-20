find_package(Clang REQUIRED CONFIG)

# FIXME: CLANG_CMAKE_DIR seems like the most stable way to find this, but
# really there is no way to reliably discover this header.
#
# We effectively back up to the Clang output directory (for the case of a build
# tree) or install prefix (for the case of an installed copy), and then search
# for a file named opencl-c.h anywhere below that. We take the first result in
# the case where there are multiple (e.g. if there is an installed copy nested
# in a build directory). This is a bit imprecise, but it covers cases like MSVC
# adding some additional configuration-specific subdirectories to the build
# tree but not to an installed copy.
file(GLOB_RECURSE OPENCL_C_H_LIST "${CLANG_CMAKE_DIR}/../../../*/opencl-c.h")
list(GET OPENCL_C_H_LIST 0 OPENCL_C_H)
if (NOT EXISTS "${OPENCL_C_H}" OR IS_DIRECTORY "${OPENCL_C_H}")
  message(FATAL_ERROR "Unable to locate opencl-c.h from the supplied Clang. The path '${CLANG_CMAKE_DIR}/../../../*' was searched.")
endif()

# Macro to create and install a custom target for generating PCH for given
# OpenCL version.
macro(GENERATE_PCH version)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opencl${version}-c.pch
    COMMAND clang -cc1 -x cl-header -triple amdgcn-amd-amdhsa-opencl
      -Werror -O3 -Dcl_khr_fp64 -Dcl_khr_fp16 -DNDEBUG -cl-std=CL${version}
      -emit-pch -o ${CMAKE_CURRENT_BINARY_DIR}/opencl${version}-c.pch
      < ${OPENCL_C_H}
    DEPENDS clang ${OPENCL_C_H}
    COMMENT "Generating opencl${version}-c.pch"
  )
  add_custom_target(opencl${version}-c.pch_target ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/opencl${version}-c.pch)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/opencl${version}-c.pch
    DESTINATION include)
endmacro()

GENERATE_PCH(1.2)
GENERATE_PCH(2.0)
