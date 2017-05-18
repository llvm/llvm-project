##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

set (CLANG_OCL_FLAGS "-Werror -x cl -Xclang -cl-std=CL2.0 -fblocks -target ${AMDGPU_TARGET_TRIPLE} -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200 -Dcl_khr_fp64 -Dcl_khr_fp16 -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics -Xclang -finclude-default-header ${CLANG_OPTIONS_APPEND}")
set (CLANG_OCL_LINK_FLAGS "-target ${AMDGPU_TARGET_TRIPLE} -mcpu=fiji")

set (LLVM_LINK "${LLVM_TOOLS_BINARY_DIR}/llvm-link")
set (LLVM_OBJDUMP "${LLVM_TOOLS_BINARY_DIR}/llvm-objdump")

set (BC_EXT .bc)
set (LIB_SUFFIX ".lib${BC_EXT}")
set (CMAKE_INCLUDE_FLAG_OCL "-I")
set (CMAKE_OCL_COMPILER_ENV_VAR OCL)
set (CMAKE_OCL_OUTPUT_EXTENTION ${BC_EXT})
set (CMAKE_OCL_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
set (CMAKE_OCL_COMPILE_OBJECT "${CMAKE_OCL_COMPILER} <INCLUDES> -o <OBJECT> <FLAGS> -c <SOURCE>")
set (CMAKE_OCL_LINK_EXECUTABLE "${LLVM_LINK} -o <TARGET> <LINK_LIBRARIES>")
set (CMAKE_OCL_CREATE_STATIC_LIBRARY "${LLVM_LINK} -o <TARGET> <OBJECTS>")
set (CMAKE_OCL_SOURCE_FILE_EXTENSIONS "cl")
set (CMAKE_EXECUTABLE_SUFFIX_OCL ".co")

macro(opencl_bc_lib name)
  set(lib_tgt ${name}_lib)
  list(APPEND AMDGCN_LIB_LIST ${lib_tgt})
  set(AMDGCN_LIB_LIST ${AMDGCN_LIB_LIST} PARENT_SCOPE)
  foreach(file ${ARGN})
    get_filename_component(fext ${file} EXT)
    #mark files as OCL source
    if (fext STREQUAL ".cl")
      set_source_files_properties(${file} PROPERTIES LANGUAGE "OCL")
    endif()
    #mark files as OCL object to add them to link
    if (fext STREQUAL ".ll")
      set_source_files_properties(${file} PROPERTIES EXTERNAL_OBJECT TRUE)
    endif()
  endforeach()

  add_library(${lib_tgt} STATIC ${ARGN})

  if(NOT ROCM_DEVICELIB_STANDALONE_BUILD)
    add_dependencies(${lib_tgt} llvm-link clang)
  endif(NOT ROCM_DEVICELIB_STANDALONE_BUILD)

  set_target_properties(${lib_tgt} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ARCHIVE_OUTPUT_NAME "${name}"
    PREFIX "" SUFFIX ${LIB_SUFFIX}
    COMPILE_FLAGS "${CLANG_OCL_FLAGS} -emit-llvm"
    LANGUAGE "OCL" LINKER_LANGUAGE "OCL")
  set(output_name "${name}.amdgcn${BC_EXT}")

  if (TARGET prepare-builtins)
    add_dependencies(${lib_tgt} prepare-builtins)
  endif()

  add_custom_command(TARGET ${lib_tgt}
    POST_BUILD
    COMMAND ${PREPARE_BUILTINS} ${name}${LIB_SUFFIX} -o ${output_name}
    DEPENDS ${lib_tgt}
    COMMENT "Generating ${output_name}"
  )
  install (FILES ${CMAKE_CURRENT_BINARY_DIR}/${output_name} DESTINATION lib COMPONENT device-libs)
endmacro(opencl_bc_lib)

macro(clang_opencl_code name dir)
  set(tgt_name ${name}_code)
  set_source_files_properties(${dir}/${name}.cl PROPERTIES LANGUAGE "OCL")
  add_executable(${tgt_name} ${dir}/${name}.cl)
  set(mlink_flags)
  foreach (lib ${ARGN})
    add_dependencies(${tgt_name} ${lib}_lib)
    get_target_property(lib_file ${lib}_lib ARCHIVE_OUTPUT_NAME)
    get_target_property(lib_path ${lib}_lib ARCHIVE_OUTPUT_DIRECTORY)
    set(mlink_flags "${mlink_flags} -Xclang -mlink-bitcode-file -Xclang ${lib_path}/${lib_file}.amdgcn${BC_EXT}")
  endforeach()
  set(CMAKE_OCL_FLAGS "${CMAKE_OCL_FLAGS} -mcpu=fiji ${mlink_flags}")
  #dummy link since clang already generated CodeObject file
  set(CMAKE_OCL_LINK_EXECUTABLE "${CMAKE_COMMAND} -E copy <OBJECTS> <TARGET>")
  set_target_properties(${tgt_name} PROPERTIES
    COMPILE_FLAGS "${CLANG_OCL_FLAGS} ${CMAKE_OCL_FLAGS}"
    LANGUAGE "OCL" LINKER_LANGUAGE "OCL")

endmacro(clang_opencl_code)

set (oclc_default_libs
  oclc_correctly_rounded_sqrt_off
  oclc_daz_opt_off
  oclc_finite_only_off
  oclc_isa_version_803
  oclc_unsafe_math_off
)

macro(clang_opencl_test name dir)
  clang_opencl_code(${name} ${dir} opencl ocml ockl ${oclc_default_libs} irif)
  add_test(
    NAME ${name}:llvm-objdump
    COMMAND ${LLVM_OBJDUMP} -disassemble -mcpu=fiji $<TARGET_FILE:${name}_code>
  )
  if(AMDHSACOD)
    add_test(
      NAME ${name}:amdhsacod
      COMMAND ${AMDHSACOD} -test -code $<TARGET_FILE:${name}_code>
    )
  endif()
endmacro(clang_opencl_test)

macro(clang_opencl_test_file dir fname)
  get_filename_component(name ${fname} NAME_WE)
  get_filename_component(fdir ${fname} DIRECTORY)
  clang_opencl_test(${name} ${dir}/${fdir})
endmacro()
