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

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20.0")
  # The policy change was for handling of relative paths for
  # DEPFILE. We only use absolute paths but cmake still feels the need
  # to complain without setting this.
  cmake_policy(SET CMP0116 NEW)
endif()


if (WIN32)
  set(EXE_SUFFIX ".exe")
else()
  set(EXE_SUFFIX)
endif()

# -Wno-error=atomic-alignment was added to workaround build problems due to
# potential mis-aligned atomic ops detected by clang
set(CLANG_OCL_FLAGS -fcolor-diagnostics -Werror -Wno-error=atomic-alignment -x cl -Xclang
  -cl-std=CL2.0 -target "${AMDGPU_TARGET_TRIPLE}" -fvisibility=protected -fomit-frame-pointer
  -Xclang -finclude-default-header -Xclang -fexperimental-strict-floating-point
  -Xclang -fdenormal-fp-math=dynamic
  -nogpulib -cl-no-stdinc "${CLANG_OPTIONS_APPEND}")

# For compatibility with the MSVC headers we use a 32-bit wchar. Users linking
# against us must also use a short wchar.
if (WIN32)
  set(CLANG_OCL_FLAGS ${CLANG_OCL_FLAGS} -fshort-wchar)
endif()

# Disable code object version module flag.
set(CLANG_OCL_FLAGS ${CLANG_OCL_FLAGS} -Xclang -mcode-object-version=none)

set (BC_EXT .bc)
set (LIB_SUFFIX ".lib${BC_EXT}")
set (STRIP_SUFFIX ".strip${BC_EXT}")
set (FINAL_SUFFIX "${BC_EXT}")
set (INSTALL_ROOT_SUFFIX "amdgcn/bitcode")

if (NOT ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_NEW STREQUAL "")
  set(INSTALL_ROOT_SUFFIX "${ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_NEW}/bitcode")
endif()

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

  # Mirror the install layout structure.
  set(OUTPUT_DIR ${PROJECT_BINARY_DIR}/${INSTALL_ROOT_SUFFIX})
  file(MAKE_DIRECTORY ${OUTPUT_DIR})

  set(OUT_NAME ${name})
  set(OUTPUT_BC_LIB ${OUTPUT_DIR}/${name}${FINAL_SUFFIX})

  set(clean_files)

  list(APPEND AMDGCN_LIB_LIST ${name})
  set(AMDGCN_LIB_LIST ${AMDGCN_LIB_LIST} PARENT_SCOPE)

  list(APPEND AMDGCN_DEP_LIST ${name})
  set(AMDGCN_DEP_LIST ${AMDGCN_DEP_LIST} PARENT_SCOPE)

  set_inc_options()
  set(deps)
  foreach(file ${OPENCL_BC_LIB_SOURCES})
    get_filename_component(fname "${file}" NAME)
    get_filename_component(fname_we "${file}" NAME_WE)
    get_filename_component(fext "${file}" EXT)
    if (fext STREQUAL ".cl")
      set(output "${CMAKE_CURRENT_BINARY_DIR}/${fname_we}${BC_EXT}")
      set(depfile "${CMAKE_CURRENT_BINARY_DIR}/${fname}.d")

      get_property(file_specific_flags SOURCE "${file}" PROPERTY COMPILE_FLAGS)

      add_custom_command(OUTPUT "${output}"
        COMMAND $<TARGET_FILE:clang> ${inc_options} ${CLANG_OCL_FLAGS}
          ${file_specific_flags}
          -emit-llvm -c "${file}" -o "${output}"
          -MD -MF ${depfile}
         MAIN_DEPENDENCY "${file}"
         DEPENDS "$<TARGET_FILE:clang>"
         DEPFILE ${depfile})
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
    "${CMAKE_CURRENT_BINARY_DIR}/${OUT_NAME}_response" @ONLY)

  add_custom_command(OUTPUT ${OUTPUT_BC_LIB}
    # Link regular library dependencies
    COMMAND $<TARGET_FILE:llvm-link>
      -o "${OUT_NAME}.link0${LIB_SUFFIX}" "@${OUT_NAME}_response"
    # Extra link step with internalize
    COMMAND $<TARGET_FILE:llvm-link> -internalize -only-needed "${name}.link0${LIB_SUFFIX}"
      -o "${OUT_NAME}${LIB_SUFFIX}" ${internal_link_libs}
    COMMAND $<TARGET_FILE:opt> -passes=amdgpu-unify-metadata,strip
      -o "${OUT_NAME}${STRIP_SUFFIX}" "${OUT_NAME}${LIB_SUFFIX}"
    COMMAND "${PREPARE_BUILTINS}"
      -o ${OUTPUT_BC_LIB} "${OUT_NAME}${STRIP_SUFFIX}"
      DEPENDS "${deps}" "${CMAKE_CURRENT_BINARY_DIR}/${OUT_NAME}_response" "${PREPARE_BUILTINS}" ${internal_link_libs})

  add_custom_target("${name}" ALL
    DEPENDS "${OUTPUT_DIR}/${OUT_NAME}${FINAL_SUFFIX}"
    SOURCES ${OPENCL_BC_LIB_SOURCES})
  set_target_properties(${name} PROPERTIES
    OUTPUT_NAME "${OUTPUT_DIR}/${OUT_NAME}${FINAL_SUFFIX}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ARCHIVE_OUTPUT_NAME "${name}"
    PREFIX "" SUFFIX ${FINAL_SUFFIX})

  list(APPEND clean_files
    "${OUT_NAME}${LIB_SUFFIX}" "${OUT_NAME}${STRIP_SUFFIX}")

  set_property(GLOBAL APPEND PROPERTY AMD_DEVICE_LIBS ${name})

  if(NOT ROCM_DEVICELIB_STANDALONE_BUILD)
    add_dependencies("${name}" llvm-link clang opt llvm-objdump)
  endif()

  if (TARGET prepare-builtins)
    add_dependencies("${name}" prepare-builtins)
  endif()

  set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${clean_files}")

  install(FILES ${OUTPUT_BC_LIB}
    DESTINATION ${INSTALL_ROOT_SUFFIX}
    COMPONENT device-libs)
endmacro()

function(clang_opencl_code name dir)
  set(TEST_TGT "${name}_code")
  set(OUT_NAME "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  set(mlink_flags)
  foreach (lib ${ARGN})
    get_target_property(lib_path "${lib}" OUTPUT_NAME)
    list(APPEND mlink_flags
      -Xclang -mlink-bitcode-file
      -Xclang "${lib_path}")
  endforeach()
  set_inc_options()
  add_custom_command(OUTPUT "${OUT_NAME}.co"
    COMMAND "$<TARGET_FILE:clang>" ${inc_options} ${CLANG_OCL_FLAGS}
      -mcpu=fiji ${mlink_flags} -o "${OUT_NAME}.co" -c "${dir}/${name}.cl"
    DEPENDS "${dir}/${name}.cl")
  add_custom_target("${TEST_TGT}" ALL
    DEPENDS "${OUT_NAME}.co"
    SOURCES "${dir}/${name}.cl")
  set_target_properties(${TEST_TGT} PROPERTIES
    OUTPUT_NAME "${OUT_NAME}.co")
  foreach (lib ${ARGN})
    add_dependencies(${TEST_TGT} ${lib})
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
    COMMAND $<TARGET_FILE:llvm-objdump> -disassemble -mcpu=fiji "${name}.co"
  )
endmacro()

macro(clang_opencl_test_file dir fname)
  get_filename_component(name ${fname} NAME_WE)
  get_filename_component(fdir ${fname} DIRECTORY)
  clang_opencl_test(${name} ${dir}/${fdir})
endmacro()
