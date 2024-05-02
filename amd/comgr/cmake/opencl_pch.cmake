if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(Clang REQUIRED CONFIG)
  execute_process(COMMAND "${CLANG_CMAKE_DIR}/../../../bin/clang" -print-resource-dir OUTPUT_VARIABLE CLANG_RESOURCE_DIR)
  string(STRIP ${CLANG_RESOURCE_DIR} CLANG_RESOURCE_DIR)
  set(OPENCL_C_H "${CLANG_RESOURCE_DIR}/include/opencl-c.h")
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
