cmake_minimum_required(VERSION 3.13.4)

# Build-time script (-P mode): reads a header manifest from trace_headers.py
# and embeds the files using embed_files() from EmbedFiles.cmake.

if(NOT GEN_LIBCXX_HEADERS_FILE)
  message(FATAL_ERROR "missing definition for GEN_LIBCXX_HEADERS_FILE")
endif()
if(NOT LIBCXX_MANIFEST_FILE)
  message(FATAL_ERROR "missing definition for LIBCXX_MANIFEST_FILE")
endif()

get_filename_component(_EMBED_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
include("${_EMBED_DIR}/EmbedFiles.cmake")

file(READ "${LIBCXX_MANIFEST_FILE}" manifest_content)
string(STRIP "${manifest_content}" manifest_content)

if("${manifest_content}" STREQUAL "")
  file(WRITE "${GEN_LIBCXX_HEADERS_FILE}"
    "#include \"comgr-libcxx-headers.h\"\n\n"
    "llvm::ArrayRef<COMGR::ResourceDirResource> COMGR::getLibcxxHeaderFiles() {\n"
    "  return {};\n}\n\n"
    "llvm::ArrayRef<COMGR::ResourceDirResource> COMGR::getClangBuiltinHeaderFiles() {\n"
    "  return {};\n}\n")
  return()
endif()

# Parse manifest into libcxx and clang file lists
set(libcxx_rel_paths)
set(libcxx_abs_paths)
set(clang_rel_paths)
set(clang_abs_paths)

string(REPLACE "\n" ";" manifest_lines "${manifest_content}")
foreach(line ${manifest_lines})
  if("${line}" STREQUAL "")
    continue()
  endif()
  string(REPLACE "\t" ";" fields "${line}")
  list(GET fields 0 type_name)
  list(GET fields 1 rel_path)
  list(GET fields 2 abs_path)

  if("${type_name}" STREQUAL "libcxx")
    list(APPEND libcxx_rel_paths "${rel_path}")
    list(APPEND libcxx_abs_paths "${abs_path}")
  elseif("${type_name}" STREQUAL "clang")
    list(APPEND clang_rel_paths "${rel_path}")
    list(APPEND clang_abs_paths "${abs_path}")
  endif()
endforeach()

# Embed both groups, saving results under prefixed names
embed_files(RELATIVE_PATHS ${libcxx_rel_paths} ABSOLUTE_PATHS ${libcxx_abs_paths}
  PREFIX comgr_libcxx OUTPUT_DIR libcxx_headers/libcxx)
foreach(var ARRAY_CONTENT EMBED_LIST SYM_DECLARATIONS OBJECT_FILES LIB_SOURCES)
  set(LIBCXX_${var} "${EMBED_${var}}")
endforeach()

embed_files(RELATIVE_PATHS ${clang_rel_paths} ABSOLUTE_PATHS ${clang_abs_paths}
  PREFIX comgr_clang_builtin OUTPUT_DIR libcxx_headers/clang)
foreach(var ARRAY_CONTENT EMBED_LIST SYM_DECLARATIONS OBJECT_FILES LIB_SOURCES)
  set(CLANG_${var} "${EMBED_${var}}")
endforeach()

# Generate the C++ source file
list(JOIN LIBCXX_ARRAY_CONTENT "\n" LIBCXX_ARRAY_CONTENT)
list(JOIN CLANG_ARRAY_CONTENT "\n" CLANG_ARRAY_CONTENT)

set(ALL_LIB_SOURCES ${LIBCXX_LIB_SOURCES} ${CLANG_LIB_SOURCES})
foreach(filename ${ALL_LIB_SOURCES})
  list(APPEND INCLUDE_CONTENT "#include \"${filename}\"")
endforeach()
list(JOIN INCLUDE_CONTENT "\n" INCLUDE_CONTENT)

list(APPEND FILE_CONTENT
  "${INCLUDE_CONTENT}\n"
  "#include \"comgr-libcxx-headers.h\"\n\n"
  "#ifdef _MSC_VER"
  "#define ALIGNATTR __declspec(align(4096))"
  "#else"
  "#define ALIGNATTR [[gnu::aligned(4096)]]"
  "#endif\n\n"
  "${LIBCXX_EMBED_LIST}" "${LIBCXX_SYM_DECLARATIONS}"
  "${CLANG_EMBED_LIST}" "${CLANG_SYM_DECLARATIONS}")

if(LIBCXX_ARRAY_CONTENT)
  list(APPEND FILE_CONTENT
    "\n\nstatic const COMGR::ResourceDirResource LibcxxHeaderFiles[] = {"
    "${LIBCXX_ARRAY_CONTENT}" "}\;"
    "llvm::ArrayRef<COMGR::ResourceDirResource> COMGR::getLibcxxHeaderFiles() {"
    "  return LibcxxHeaderFiles\;" "}")
else()
  list(APPEND FILE_CONTENT
    "\nllvm::ArrayRef<COMGR::ResourceDirResource> COMGR::getLibcxxHeaderFiles() {"
    "  return {}\;" "}")
endif()

if(CLANG_ARRAY_CONTENT)
  list(APPEND FILE_CONTENT
    "\n\nstatic const COMGR::ResourceDirResource ClangBuiltinHeaderFiles[] = {"
    "${CLANG_ARRAY_CONTENT}" "}\;"
    "llvm::ArrayRef<COMGR::ResourceDirResource> COMGR::getClangBuiltinHeaderFiles() {"
    "  return ClangBuiltinHeaderFiles\;" "}")
else()
  list(APPEND FILE_CONTENT
    "\nllvm::ArrayRef<COMGR::ResourceDirResource> COMGR::getClangBuiltinHeaderFiles() {"
    "  return {}\;" "}")
endif()

list(JOIN FILE_CONTENT "\n" FILE_CONTENT)
file(WRITE "${GEN_LIBCXX_HEADERS_FILE}" "${FILE_CONTENT}\n")

set(ALL_OBJECT_FILES ${LIBCXX_OBJECT_FILES} ${CLANG_OBJECT_FILES})
if(ALL_OBJECT_FILES)
  execute_process(COMMAND ${CMAKE_AR} rcs
    libcxx_headers.a ${ALL_OBJECT_FILES}
    COMMAND_ERROR_IS_FATAL ANY)
endif()
