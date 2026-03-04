cmake_minimum_required(VERSION 3.13.4)

if(NOT GEN_RESOURCE_DIR_FILE)
  message(FATAL_ERROR "missing definition for GEN_RESOURCE_DIR_FILE")
endif()

if(NOT CLANG_RESOURCE_DIR)
  message(FATAL_ERROR "missing definition for CLANG_RESOURCE_DIR")
endif()

make_directory(resource_dir)

# Keep in sync with DeviceLibs.cmake
file(GLOB_RECURSE files
     RELATIVE ${CLANG_RESOURCE_DIR}
     LIST_DIRECTORIES false
     "${CLANG_RESOURCE_DIR}/lib/amd*/*.bc"
     "${CLANG_RESOURCE_DIR}/lib/amd*/*.a")

foreach(file ${files})
  string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" sanitized_name "${file}")

  # TODO: Avoid duplicating content for symlinks
  set(const_array_name "comgr_resource_dir_${sanitized_name}")
  set(output_c_file resource_dir/${const_array_name}.inc)

  set(output_o_file resource_dir/${const_array_name}.o)
  set(output_s_file resource_dir/${const_array_name}.s)

  file(REAL_PATH ${file} file_absolute
       BASE_DIRECTORY ${CLANG_RESOURCE_DIR})

  if(COMGR_USE_EMBED)
    list(APPEND generated_embed_list
    "ALIGNATTR static constexpr unsigned char ${const_array_name}[] = {
      #embed \"${file_absolute}\" suffix(,)
      0
    }\;
    constexpr size_t ${const_array_name}_size = sizeof(${const_array_name}) - 1\;")

    list(APPEND RESOURCE_ARRAY_CONTENT
      " { \"${file}\", llvm::StringRef(reinterpret_cast<const char *>(${const_array_name}), ${const_array_name}_size)},")
  elseif(COMGR_USE_INCBIN)
    # Use .incbin assembler directive

    if(APPLE)
      set(section_directive ".section __DATA,__const")
      set(asm_prefix "_")
    else()
      set(section_directive ".section .rodata")
      set(asm_prefix "")
    endif()

    # Null terminate the file and align to 4096
    file(WRITE ${output_s_file}
      "${section_directive}
      .global ${asm_prefix}_binary_${sanitized_name}_start
      .global ${asm_prefix}_binary_${sanitized_name}_end
      .p2align 12

      ${asm_prefix}_binary_${sanitized_name}_start:
      .incbin \"${file_absolute}\"
      .byte 0
      ${asm_prefix}_binary_${sanitized_name}_end:")

    execute_process(
      COMMAND ${CMAKE_ASM_COMPILER} -c ${output_s_file} -o ${output_o_file}
      RESULT_VARIABLE asm_res
      OUTPUT_VARIABLE asm_out
      ERROR_VARIABLE asm_err)

    if(NOT asm_res EQUAL 0)
      message(FATAL_ERROR "Assembler failed:\n${asm_err}")
    endif()

    list(APPEND binary_sym_declarations
       "[[gnu::aligned(4096)]] extern const char _binary_${sanitized_name}_start[]\;"
       "extern const char _binary_${sanitized_name}_end[]\;")

    list(APPEND resource_directory_object_files ${output_o_file})
    list(APPEND RESOURCE_ARRAY_CONTENT
      " { \"${file}\", llvm::StringRef(_binary_${sanitized_name}_start, static_cast<size_t>(_binary_${sanitized_name}_end - _binary_${sanitized_name}_start) - 1)},")
    list(APPEND resource_directory_object_files ${output_o_file})
  else()
    execute_process(
      COMMAND ${BC2H_BINARY} ${file_absolute} ${output_c_file} comgr_resource_dir_${sanitized_name}
      COMMAND_ERROR_IS_FATAL ANY)
    list(APPEND generated_lib_sources ${output_c_file})
    list(APPEND RESOURCE_ARRAY_CONTENT
      " { \"${file}\", llvm::StringRef(reinterpret_cast<const char*>(${const_array_name}), ${const_array_name}_size) },")
  endif()

  # TODO: If not cross compiling, and the built clang supports the
  # host architecture, use it to perform the embed
  #
  # TODO: Use rc on Windows.
endforeach()

list(JOIN RESOURCE_ARRAY_CONTENT "\n" RESOURCE_ARRAY_CONTENT)

list(APPEND RESOURCE_DIR_INC_FILE_CONTENT
  "#include \"comgr-resource-directory.h\"\n\n"
  "#ifdef _MSC_VER"
  "#define ALIGNATTR __declspec(align(4096))"
  "#else"
  "#define ALIGNATTR [[gnu::aligned(4096)]]"
  "#endif\n\n"
  "${generated_embed_list}"
  "${binary_sym_declarations}")

if(RESOURCE_ARRAY_CONTENT)
  list(APPEND RESOURCE_DIR_INC_FILE_CONTENT
    "\n\nstatic const COMGR::ResourceDirResource ResourceDirectoryFiles[] = {"
    "${RESOURCE_ARRAY_CONTENT}"
    "}\;"
    "llvm::ArrayRef<COMGR::ResourceDirResource> COMGR::getResourceDirectoryFiles() {"
    "  return ResourceDirectoryFiles\;"
    "}")
else()
  list(APPEND RESOURCE_DIR_INC_FILE_CONTENT
    "llvm::ArrayRef<COMGR::ResourceDirResource> COMGR::getResourceDirectoryFiles() {"
    "  return {}\;"
    "}")
endif()


list(JOIN RESOURCE_DIR_INC_FILE_CONTENT "\n" RESOURCE_DIR_INC_FILE_CONTENT)

foreach(filename ${generated_lib_sources})
  list(APPEND INCLUDE_FILE_CONTENT "#include \"${filename}\"")
endforeach()

list(JOIN INCLUDE_FILE_CONTENT "\n" INCLUDE_FILE_CONTENT)

file(WRITE ${GEN_RESOURCE_DIR_FILE}
     "${INCLUDE_FILE_CONTENT}\n${RESOURCE_DIR_INC_FILE_CONTENT}\n")

if(resource_directory_object_files)
  execute_process(COMMAND ${CMAKE_AR} rcs
    resource_directory.a ${resource_directory_object_files}
    COMMAND_ERROR_IS_FATAL ANY)
endif()
