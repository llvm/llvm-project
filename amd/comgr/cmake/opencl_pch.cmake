if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
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
else()
  get_target_property(clang_build_header_dir clang-resource-headers RUNTIME_OUTPUT_DIRECTORY)
  set(OPENCL_C_H "${clang_build_header_dir}/opencl-c.h")
endif()

# Macro to create and install a custom target for generating PCH for given
# OpenCL version.
function(generate_pch version)
  # You can use add_dependencies to separately add dependencies to a
  # target, but as far as I can tell, you can't use it to add file
  # dependencies, so we have to build the different dependencies
  # lists before add_custom_command
  if(TARGET clang-resource-headers)
    # clang is an imported target in a standalone build, but the
    # generated headers are not.
    set(clang_resource_headers_gen clang-resource-headers)
  endif()

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opencl${version}-c.pch
    COMMAND $<TARGET_FILE:clang> -cc1 -x cl-header -triple amdgcn-amd-amdhsa
    -Werror -O3 -Dcl_khr_fp64 -Dcl_khr_fp16 -DNDEBUG -cl-std=CL${version}
      -emit-pch -o ${CMAKE_CURRENT_BINARY_DIR}/opencl${version}-c.pch
      < ${OPENCL_C_H}
    DEPENDS clang ${OPENCL_C_H} ${clang_resource_headers_gen}
    COMMENT "Generating opencl${version}-c.pch")

  add_custom_target(opencl${version}-c.pch_target ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/opencl${version}-c.pch)
endfunction()

generate_pch(1.2)
generate_pch(2.0)
