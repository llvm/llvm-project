find_package(Clang REQUIRED CONFIG)

# FIXME: CLANG_CMAKE_DIR seems like the most stable way to find this, but
# really there is no way to reliably discover this header.
file(GLOB_RECURSE OPENCL_C_H "${CLANG_CMAKE_DIR}/../../*/opencl-c.h")

# Macro to create and install a custom target for generating PCH for given
# OpenCL version.
macro(GENERATE_PCH version)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opencl${version}-c.pch
    COMMAND clang -cc1 -x cl-header -triple amdgcn-amd-amdhsa-opencl
      -Werror -O3 -Dcl_khr_fp64 -Dcl_khr_fp16 -DNDEBUG -cl-std=CL${version}
      -emit-pch -o ${CMAKE_CURRENT_BINARY_DIR}/opencl${version}-c.pch
      ${OPENCL_C_H}
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
