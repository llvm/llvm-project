if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(Clang REQUIRED CONFIG)

  # FIXME: CLANG_CMAKE_DIR seems like the most stable way to find this, but
  # really there is no way to reliably discover this header.
  #
  # We effectively back up to the Clang output directory (for the case of a build
  # tree) or install prefix (for the case of an installed copy), and then search
  # for a file named opencl-c-base.h anywhere below that. We take the first result in
  # the case where there are multiple (e.g. if there is an installed copy nested
  # in a build directory). This is a bit imprecise, but it covers cases like MSVC
  # adding some additional configuration-specific subdirectories to the build
  # tree but not to an installed copy.
  file(GLOB_RECURSE OPENCL_C_H_LIST "${CLANG_CMAKE_DIR}/../../../*/opencl-c-base.h")

  list(GET OPENCL_C_H_LIST 0 OPENCL_C_H)

  if (NOT EXISTS "${OPENCL_C_H}" OR IS_DIRECTORY "${OPENCL_C_H}")
    message(FATAL_ERROR "Unable to locate opencl-c-base.h from the supplied Clang. The path '${CLANG_CMAKE_DIR}/../../../*' was searched.")
  endif()
else()
  get_target_property(clang_build_header_dir clang-resource-headers RUNTIME_OUTPUT_DIRECTORY)
  set(OPENCL_C_H "${clang_build_header_dir}/opencl-c-base.h")
endif()
