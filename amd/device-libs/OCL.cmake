##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

# Required because we need to generate response files on windows for long
# command-lines, but the only way to do this as part of the dependency graph is
# configure_file and we are included from multiple places. To get around this
# we `file(WRITE)` a file with an @variable reference and `configure_file` it.
cmake_policy(SET CMP0053 OLD)

if (WIN32)
  set(EXE_SUFFIX ".exe")
else()
  set(EXE_SUFFIX)
endif()
set(CLANG "${LLVM_TOOLS_BINARY_DIR}/clang${EXE_SUFFIX}")
set(LLVM_LINK "${LLVM_TOOLS_BINARY_DIR}/llvm-link${EXE_SUFFIX}")
set(LLVM_OBJDUMP "${LLVM_TOOLS_BINARY_DIR}/llvm-objdump${EXE_SUFFIX}")
set(LLVM_OPT "${LLVM_TOOLS_BINARY_DIR}/opt${EXE_SUFFIX}")

# -Wno-error=atomic-alignment was added to workaround build problems due to
# potential mis-aligned atomic ops detected by clang
set(CLANG_OCL_FLAGS -Werror -Wno-error=atomic-alignment -x cl -Xclang
  -cl-std=CL2.0 -target "${AMDGPU_TARGET_TRIPLE}" -fvisibility=protected
  -Xclang -finclude-default-header "${CLANG_OPTIONS_APPEND}")

set (BC_EXT .bc)
set (LIB_SUFFIX ".lib${BC_EXT}")
set (STRIP_SUFFIX ".strip${BC_EXT}")
set (FINAL_SUFFIX ".amdgcn${BC_EXT}")

# Set `inc_options` to contain Clang command-line for include directories for
# current source directory.
macro(set_inc_options)
  get_property(inc_dirs
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PROPERTY INCLUDE_DIRECTORIES)
  set(inc_options)
  foreach(inc_dir ${inc_dirs})
    list(APPEND inc_options "-I${inc_dir}")
  endforeach()
endmacro()

# called with NAME: library name
#             SOURCES: .cl and .ll source files
#             INTERNAL_LINK_LIBS: Extra .lls to be linked and internalized into final library
macro(opencl_bc_lib)
  set(parse_options)
  set(one_value_args NAME)
  set(multi_value_args SOURCES INTERNAL_LINK_LIBS)

  cmake_parse_arguments(OPENCL_BC_LIB "${parse_options}" "${one_value_args}"
                                      "${multi_value_args}" ${ARGN})

  set(name ${OPENCL_BC_LIB_NAME})
  set(sources ${OPENCL_BC_LIB_SOURCES})
  set(internal_link_libs ${OPENCL_BC_LIB_INTERNAL_LINK_LIBS})

  get_target_property(irif_lib_output irif_lib OUTPUT_NAME)

  set(OUT_NAME "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  set(LIB_TGT ${name}_lib)
  set(clean_files)

  list(APPEND AMDGCN_LIB_LIST ${LIB_TGT})
  set(AMDGCN_LIB_LIST ${AMDGCN_LIB_LIST} PARENT_SCOPE)

  list(APPEND AMDGCN_DEP_LIST ${LIB_TGT})
  set(AMDGCN_DEP_LIST ${AMDGCN_DEP_LIST} PARENT_SCOPE)

  set_inc_options()
  set(deps)
  foreach(file ${OPENCL_BC_LIB_SOURCES})
    get_filename_component(fname_we "${file}" NAME_WE)
    get_filename_component(fext "${file}" EXT)
    if (fext STREQUAL ".cl")
      set(output "${CMAKE_CURRENT_BINARY_DIR}/${fname_we}${BC_EXT}")
      add_custom_command(OUTPUT "${output}"
        COMMAND "${CLANG}" ${inc_options} ${CLANG_OCL_FLAGS}
          -emit-llvm -Xclang -mlink-builtin-bitcode -Xclang "${irif_lib_output}"
          -c "${file}" -o "${output}"
        DEPENDS "${file}" "${irif_lib_output}" "${CLANG}")
      list(APPEND deps "${output}")
      list(APPEND clean_files "${output}")
    endif()
    if (fext STREQUAL ".ll")
      list(APPEND deps "${file}")
    endif()
  endforeach()

  # The llvm-link command-lines can get long enough to trigger strange behavior
  # on Windows. LLVM tools support "response files" which can work around this:
  # http://llvm.org/docs/CommandLine.html#response-files
  set(RESPONSE_COMMAND_LINE)
  foreach(dep ${deps})
    set(RESPONSE_COMMAND_LINE "${RESPONSE_COMMAND_LINE} ${dep}")
  endforeach()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/response.in" "@RESPONSE_COMMAND_LINE@")
  configure_file("${CMAKE_CURRENT_BINARY_DIR}/response.in"
    "${OUT_NAME}_response" @ONLY)

  add_custom_command(OUTPUT "${OUT_NAME}${FINAL_SUFFIX}"
    # Link regular library dependencies
    COMMAND "${LLVM_LINK}"
      -o "${OUT_NAME}.link0${LIB_SUFFIX}" "@${OUT_NAME}_response"
    # Extra link step with internalize
    COMMAND "${LLVM_LINK}" -internalize -only-needed "${OUT_NAME}.link0${LIB_SUFFIX}"
      -o "${OUT_NAME}${LIB_SUFFIX}" ${internal_link_libs}
    COMMAND "${LLVM_OPT}" -strip
      -o "${OUT_NAME}${STRIP_SUFFIX}" "${OUT_NAME}${LIB_SUFFIX}"
    COMMAND "${PREPARE_BUILTINS}"
      -o "${OUT_NAME}${FINAL_SUFFIX}" "${OUT_NAME}${STRIP_SUFFIX}"
    DEPENDS "${deps}" "${OUT_NAME}_response" "${PREPARE_BUILTINS}" ${internal_link_libs})

  add_custom_target("${LIB_TGT}" ALL
    DEPENDS "${OUT_NAME}${FINAL_SUFFIX}"
    SOURCES ${OPENCL_BC_LIB_SOURCES})
  set_target_properties(${LIB_TGT} PROPERTIES
    OUTPUT_NAME "${OUT_NAME}${FINAL_SUFFIX}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ARCHIVE_OUTPUT_NAME "${name}"
    PREFIX "" SUFFIX ${FINAL_SUFFIX})

  list(APPEND clean_files
    "${OUT_NAME}${LIB_SUFFIX}" "${OUT_NAME}${STRIP_SUFFIX}")

  if(NOT ROCM_DEVICELIB_STANDALONE_BUILD)
    add_dependencies("${LIB_TGT}" llvm-link clang opt llvm-objdump)
  endif()

  if (TARGET prepare-builtins)
    add_dependencies("${LIB_TGT}" prepare-builtins)
  endif()
  add_dependencies("${LIB_TGT}" irif_lib)

  set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${clean_files}")

  install(FILES "${OUT_NAME}${FINAL_SUFFIX}"
    DESTINATION lib
    COMPONENT device-libs)
endmacro()

function(clang_opencl_code name dir)
  set(TEST_TGT "${name}_code")
  set(OUT_NAME "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  set(mlink_flags)
  foreach (lib ${ARGN})
    get_target_property(lib_path "${lib}_lib" OUTPUT_NAME)
    list(APPEND mlink_flags
      -Xclang -mlink-bitcode-file
      -Xclang "${lib_path}")
  endforeach()
  set_inc_options()
  add_custom_command(OUTPUT "${OUT_NAME}.co"
    COMMAND "${CLANG}" ${inc_options} ${CLANG_OCL_FLAGS}
      -mcpu=fiji ${mlink_flags} -o "${OUT_NAME}.co" -c "${dir}/${name}.cl"
    DEPENDS "${dir}/${name}.cl")
  add_custom_target("${TEST_TGT}" ALL
    DEPENDS "${OUT_NAME}.co"
    SOURCES "${dir}/${name}.cl")
  set_target_properties(${TEST_TGT} PROPERTIES
    OUTPUT_NAME "${OUT_NAME}.co")
  foreach (lib ${ARGN})
    add_dependencies(${TEST_TGT} ${lib}_lib)
  endforeach()
endfunction()

set(OCLC_DEFAULT_LIBS
  oclc_correctly_rounded_sqrt_off
  oclc_daz_opt_off
  oclc_finite_only_off
  oclc_isa_version_803
  oclc_unsafe_math_off)

macro(clang_opencl_test name dir)
  clang_opencl_code(${name} ${dir} hip opencl ocml ockl ${OCLC_DEFAULT_LIBS})
  add_test(
    NAME ${name}:llvm-objdump
    COMMAND ${LLVM_OBJDUMP} -disassemble -mcpu=fiji "${name}.co"
  )
endmacro()

macro(clang_opencl_test_file dir fname)
  get_filename_component(name ${fname} NAME_WE)
  get_filename_component(fdir ${fname} DIRECTORY)
  clang_opencl_test(${name} ${dir}/${fdir})
endmacro()
