##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

set (LLVM_LINK "${LLVM_TOOLS_BINARY_DIR}/llvm-link")
set (LLVM_OBJDUMP "${LLVM_TOOLS_BINARY_DIR}/llvm-objdump")

set (BC_EXT .amdgcn.bc)
set (CMAKE_OCL_COMPILER_ENV_VAR OCLC)
set (CMAKE_OCL_OUTPUT_EXTENTION .bc)
set (CMAKE_OCL_OUTPUT_EXTENTION_REPLACE 1)
set (CMAKE_OCL_COMPILER ${CMAKE_C_COMPILER})
set (CMAKE_OCL_COMPILE_OBJECT "<CMAKE_OCL_COMPILER> -o <OBJECT> <FLAGS> -c <SOURCE>")
set (CMAKE_OCL_LINK_EXECUTABLE "${LLVM_LINK} -o <TARGET> <LINK_LIBRARIES>")
set (CMAKE_OCL_CREATE_STATIC_LIBRARY "${LLVM_LINK} -o <TARGET> <OBJECTS>")
set (CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "")
set (CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "")
set (CMAKE_C_IMPLICIT_LINK_LIBRARIES "")
set (CMAKE_C_IMPLICIT_LINK_DIRECTORIES "")
set (CMAKE_OCL_IMPLICIT_LINK_LIBRARIES "")
set (CMAKE_OCL_IMPLICIT_LINK_DIRECTORIES "")
set (CMAKE_OCL_LINKER_PREFERENCE_PROPAGATES 0)
set (CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

macro(clang_csources name dir)
  set(csources)
  foreach(file ${ARGN})
    file(RELATIVE_PATH rfile ${dir} ${file})
    get_filename_component(rdir ${rfile} DIRECTORY)
    get_filename_component(fname ${rfile} NAME_WE)
    get_filename_component(fext ${rfile} EXT)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${rdir})
    if (fext STREQUAL ".cl")
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.c)
      add_custom_command(
        OUTPUT ${cfile}
        COMMAND cp ${file} ${cfile}
        DEPENDS ${file}
      )
      list(APPEND csources ${cfile})
    endif()
    if (fext STREQUAL ".ll")
      list(APPEND csources ${file})
      set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${rdir}/${fname}.o)
      add_custom_command(
        OUTPUT ${cfile}
        COMMAND cp ${file} ${cfile}
        DEPENDS ${file}
      )
      list(APPEND csources ${cfile})
    endif()
  endforeach()
endmacro()

macro(clang_opencl_bc_lib name dir)
  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  clang_csources(${name} ${dir} ${ARGN})
  add_library(${name}_lib_bc STATIC ${csources})
  set_target_properties(${name}_lib_bc PROPERTIES OUTPUT_NAME ${name})
  set_target_properties(${name}_lib_bc PROPERTIES PREFIX "" SUFFIX ".lib.bc")
  set_target_properties(${name}_lib_bc PROPERTIES COMPILE_FLAGS "${CLANG_OCL_FLAGS} -emit-llvm")
  set_target_properties(${name}_lib_bc PROPERTIES LANGUAGE OCL)
  set_target_properties(${name}_lib_bc PROPERTIES LINKER_LANGUAGE OCL)
endmacro(clang_opencl_bc_lib)

macro(prepare_builtins name)
  add_custom_command(
    OUTPUT ${name}${BC_EXT}
    COMMAND $<TARGET_FILE:prepare-builtins> ${name}.lib.bc -o ${name}${BC_EXT}
    DEPENDS prepare-builtins ${name}_lib_bc
  )
  add_custom_target(${name}_bc ALL
    DEPENDS ${name}${BC_EXT}
  )
  set(TARGET_FILE_${name} ${CMAKE_CURRENT_BINARY_DIR}/${name}${BC_EXT} CACHE INTERNAL "")
endmacro(prepare_builtins)

macro(clang_opencl_bc_builtins_lib name dir)
  clang_opencl_bc_lib(${name} ${dir} ${ARGN})
  prepare_builtins(${name})
  install (FILES ${CMAKE_CURRENT_BINARY_DIR}/${name}${BC_EXT} DESTINATION lib)
endmacro(clang_opencl_bc_builtins_lib)

macro(clang_opencl_code name dir)
  clang_csources(${name}_code ${dir} ${dir}/${name}.cl)
  add_executable(${name}_code ${csources})
  set(mlink_flags)
  foreach (lib ${ARGN})
    add_dependencies(${name}_code ${lib}_bc)
    set(mlink_flags "${mlink_flags} -Xclang -mlink-bitcode-file -Xclang ${TARGET_FILE_${lib}}")
  endforeach()
  set_target_properties(${name}_code PROPERTIES LINKER_LANGUAGE C)
  set_target_properties(${name}_code PROPERTIES OUTPUT_NAME ${name})
  set_target_properties(${name}_code PROPERTIES COMPILE_FLAGS "${CLANG_OCL_FLAGS} ${CLANG_OCL_LINK_FLAGS} ${mlink_flags}")
  set_target_properties(${name}_code PROPERTIES LINK_FLAGS "${CLANG_OCL_LINK_FLAGS}")
  set_target_properties(${name}_code PROPERTIES PREFIX "" SUFFIX ".co")
endmacro(clang_opencl_code)

enable_testing()

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
  get_filename_component(fext ${fname} EXT)
  get_filename_component(name ${fname} NAME_WE)
  get_filename_component(fdir ${fname} DIRECTORY)
  clang_opencl_test(${name} ${dir}/${fdir})
endmacro()
