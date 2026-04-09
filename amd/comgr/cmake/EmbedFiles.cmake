# EmbedFiles.cmake - Shared file embedding function for comgr.
# Supports #embed, .incbin, and bc2h. See EmbedResourceDir.cmake for usage.

function(embed_files)
  cmake_parse_arguments(ARG "" "PREFIX;OUTPUT_DIR" "RELATIVE_PATHS;ABSOLUTE_PATHS" ${ARGN})

  make_directory(${ARG_OUTPUT_DIR})

  set(_array_content)
  set(_embed_list)
  set(_sym_declarations)
  set(_object_files)
  set(_lib_sources)

  list(LENGTH ARG_RELATIVE_PATHS _num_files)
  if(_num_files EQUAL 0)
    foreach(var ARRAY_CONTENT EMBED_LIST SYM_DECLARATIONS OBJECT_FILES LIB_SOURCES)
      set(EMBED_${var} "" PARENT_SCOPE)
    endforeach()
    return()
  endif()

  math(EXPR _last_idx "${_num_files} - 1")

  foreach(idx RANGE ${_last_idx})
    list(GET ARG_RELATIVE_PATHS ${idx} rel_path)
    list(GET ARG_ABSOLUTE_PATHS ${idx} abs_path)

    string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" sanitized_name "${rel_path}")
    set(const_array_name "${ARG_PREFIX}_${sanitized_name}")
    set(output_c_file "${ARG_OUTPUT_DIR}/${const_array_name}.inc")
    set(output_o_file "${ARG_OUTPUT_DIR}/${const_array_name}.o")
    set(output_s_file "${ARG_OUTPUT_DIR}/${const_array_name}.s")

    if(COMGR_USE_EMBED)
      list(APPEND _embed_list
      "ALIGNATTR static constexpr unsigned char ${const_array_name}[] = {
        #embed \"${abs_path}\" suffix(,)
        0
      }\;
      constexpr size_t ${const_array_name}_size = sizeof(${const_array_name}) - 1\;")

      list(APPEND _array_content
        " { \"${rel_path}\", llvm::StringRef(reinterpret_cast<const char *>(${const_array_name}), ${const_array_name}_size)},")

    elseif(COMGR_USE_INCBIN)
      if(APPLE)
        set(section_directive ".section __DATA,__const")
        set(asm_prefix "_")
      else()
        set(section_directive ".section .rodata")
        set(asm_prefix "")
      endif()

      file(WRITE "${output_s_file}"
        "${section_directive}
        .global ${asm_prefix}_binary_${sanitized_name}_start
        .global ${asm_prefix}_binary_${sanitized_name}_end
        .p2align 12

        ${asm_prefix}_binary_${sanitized_name}_start:
        .incbin \"${abs_path}\"
        .byte 0
        ${asm_prefix}_binary_${sanitized_name}_end:")

      execute_process(
        COMMAND ${CMAKE_ASM_COMPILER} -c "${output_s_file}" -o "${output_o_file}"
        RESULT_VARIABLE asm_res
        OUTPUT_VARIABLE asm_out
        ERROR_VARIABLE asm_err)

      if(NOT asm_res EQUAL 0)
        message(FATAL_ERROR "Assembler failed:\n${asm_err}")
      endif()

      list(APPEND _sym_declarations
         "[[gnu::aligned(4096)]] extern const char _binary_${sanitized_name}_start[]\;"
         "extern const char _binary_${sanitized_name}_end[]\;")

      list(APPEND _object_files "${output_o_file}")
      list(APPEND _array_content
        " { \"${rel_path}\", llvm::StringRef(_binary_${sanitized_name}_start, static_cast<size_t>(_binary_${sanitized_name}_end - _binary_${sanitized_name}_start) - 1)},")

    else()
      execute_process(
        COMMAND ${BC2H_BINARY} "${abs_path}" "${output_c_file}" "${const_array_name}"
        COMMAND_ERROR_IS_FATAL ANY)
      list(APPEND _lib_sources "${output_c_file}")
      list(APPEND _array_content
        " { \"${rel_path}\", llvm::StringRef(reinterpret_cast<const char*>(${const_array_name}), ${const_array_name}_size) },")
    endif()
  endforeach()

  set(EMBED_ARRAY_CONTENT "${_array_content}" PARENT_SCOPE)
  set(EMBED_EMBED_LIST "${_embed_list}" PARENT_SCOPE)
  set(EMBED_SYM_DECLARATIONS "${_sym_declarations}" PARENT_SCOPE)
  set(EMBED_OBJECT_FILES "${_object_files}" PARENT_SCOPE)
  set(EMBED_LIB_SOURCES "${_lib_sources}" PARENT_SCOPE)
endfunction()
