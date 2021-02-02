# A rule for self contained header file targets.
# This rule merely copies the header file from the current source directory to
# the current binary directory.
# Usage:
#     add_header(
#       <target name>
#       HDR <header file>
#     )
function(add_header target_name)
  cmake_parse_arguments(
    "ADD_HEADER"
    ""    # No optional arguments
    "HDR" # Single value arguments
    "DEPENDS"
    ${ARGN}
  )
  if(NOT ADD_HEADER_HDR)
    message(FATAL_ERROR "'add_header' rules requires the HDR argument specifying a headef file.")
  endif()

  set(dest_file ${CMAKE_CURRENT_BINARY_DIR}/${ADD_HEADER_HDR})
  set(src_file ${CMAKE_CURRENT_SOURCE_DIR}/${ADD_HEADER_HDR})

  add_custom_command(
    OUTPUT ${dest_file}
    COMMAND cp ${src_file} ${dest_file}
    DEPENDS ${src_file}
  )

  get_fq_target_name(${target_name} fq_target_name)
  add_custom_target(
    ${fq_target_name}
    DEPENDS ${dest_file}
  )

  if(ADD_HEADER_DEPENDS)
    get_fq_deps_list(fq_deps_list ${ADD_HEADER_DEPENDS})
    add_dependencies(
      ${fq_target_name} ${fq_deps_list}
    )
  endif()
endfunction(add_header)

# A rule for generated header file targets.
# Usage:
#     add_gen_header(
#       <target name>
#       DEF_FILE <.h.def file>
#       GEN_HDR <generated header file name>
#       PARAMS <list of name=value pairs>
#       DATA_FILES <list input data files>
#     )
function(add_gen_header target_name)
  cmake_parse_arguments(
    "ADD_GEN_HDR"
    "" # No optional arguments
    "DEF_FILE;GEN_HDR" # Single value arguments
    "PARAMS;DATA_FILES;DEPENDS"     # Multi value arguments
    ${ARGN}
  )
  if(NOT ADD_GEN_HDR_DEF_FILE)
    message(FATAL_ERROR "`add_gen_hdr` rule requires DEF_FILE to be specified.")
  endif()
  if(NOT ADD_GEN_HDR_GEN_HDR)
    message(FATAL_ERROR "`add_gen_hdr` rule requires GEN_HDR to be specified.")
  endif()

  set(out_file ${CMAKE_CURRENT_BINARY_DIR}/${ADD_GEN_HDR_GEN_HDR})
  set(in_file ${CMAKE_CURRENT_SOURCE_DIR}/${ADD_GEN_HDR_DEF_FILE})

  set(fq_data_files "")
  if(ADD_GEN_HDR_DATA_FILES)
    foreach(data_file IN LISTS ADD_GEN_HDR_DATA_FILES)
      list(APPEND fq_data_files "${CMAKE_CURRENT_SOURCE_DIR}/${data_file}")
    endforeach(data_file)
  endif()

  set(replacement_params "")
  if(ADD_GEN_HDR_PARAMS)
    list(APPEND replacement_params "--args" ${ADD_GEN_HDR_PARAMS})
  endif()

  set(gen_hdr_script "${LIBC_BUILD_SCRIPTS_DIR}/gen_hdr.py")

  file(GLOB td_includes ${LIBC_SOURCE_DIR}/spec/*.td)

  set(ENTRYPOINT_NAME_LIST_ARG ${TARGET_ENTRYPOINT_NAME_LIST})
  list(TRANSFORM ENTRYPOINT_NAME_LIST_ARG PREPEND "--e=")

  add_custom_command(
    OUTPUT ${out_file}
    COMMAND ${LIBC_TABLEGEN_EXE} -o ${out_file} --header ${ADD_GEN_HDR_GEN_HDR}
            --def ${in_file} ${replacement_params} -I ${LIBC_SOURCE_DIR}
           ${ENTRYPOINT_NAME_LIST_ARG}
           ${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/api.td

    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${in_file} ${fq_data_files} ${td_includes}
            ${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/api.td
            ${LIBC_TABLEGEN_EXE} ${LIBC_TABLEGEN_TARGET}
  )

  get_fq_target_name(${target_name} fq_target_name)
  if(ADD_GEN_HDR_DEPENDS)
    get_fq_deps_list(fq_deps_list ${ADD_GEN_HDR_DEPENDS})
  endif()
  add_custom_target(
    ${fq_target_name}
    DEPENDS ${out_file} ${fq_deps_list}
  )
endfunction(add_gen_header)
