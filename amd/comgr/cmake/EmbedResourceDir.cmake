cmake_minimum_required(VERSION 3.13.4)

if(NOT GEN_RESOURCE_DIR_FILE)
  message(FATAL_ERROR "missing definition for GEN_RESOURCE_DIR_FILE")
endif()

if(NOT CLANG_RESOURCE_DIR)
  message(FATAL_ERROR "missing definition for CLANG_RESOURCE_DIR")
endif()

get_filename_component(_EMBED_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
include("${_EMBED_DIR}/EmbedFiles.cmake")

# Keep in sync with DeviceLibs.cmake
file(GLOB_RECURSE files
     RELATIVE ${CLANG_RESOURCE_DIR}
     LIST_DIRECTORIES false
     "${CLANG_RESOURCE_DIR}/lib/amd*/*.bc"
     "${CLANG_RESOURCE_DIR}/lib/amd*/*.a")

set(rel_paths)
set(abs_paths)
foreach(file ${files})
  file(REAL_PATH ${file} file_absolute
       BASE_DIRECTORY ${CLANG_RESOURCE_DIR})
  list(APPEND rel_paths "${file}")
  list(APPEND abs_paths "${file_absolute}")
endforeach()

embed_files(
  RELATIVE_PATHS ${rel_paths}
  ABSOLUTE_PATHS ${abs_paths}
  PREFIX comgr_resource_dir
  OUTPUT_DIR resource_dir
)

list(JOIN EMBED_ARRAY_CONTENT "\n" EMBED_ARRAY_CONTENT)

list(APPEND RESOURCE_DIR_INC_FILE_CONTENT
  "#include \"comgr-resource-directory.h\"\n\n"
  "#ifdef _MSC_VER"
  "#define ALIGNATTR __declspec(align(4096))"
  "#else"
  "#define ALIGNATTR [[gnu::aligned(4096)]]"
  "#endif\n\n"
  "${EMBED_EMBED_LIST}"
  "${EMBED_SYM_DECLARATIONS}")

if(EMBED_ARRAY_CONTENT)
  list(APPEND RESOURCE_DIR_INC_FILE_CONTENT
    "\n\nstatic const COMGR::ResourceDirResource ResourceDirectoryFiles[] = {"
    "${EMBED_ARRAY_CONTENT}"
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

foreach(filename ${EMBED_LIB_SOURCES})
  list(APPEND INCLUDE_FILE_CONTENT "#include \"${filename}\"")
endforeach()

list(JOIN INCLUDE_FILE_CONTENT "\n" INCLUDE_FILE_CONTENT)

file(WRITE ${GEN_RESOURCE_DIR_FILE}
     "${INCLUDE_FILE_CONTENT}\n${RESOURCE_DIR_INC_FILE_CONTENT}\n")

if(resource_directory_object_files AND RESOURCE_DIRECTORY_ARCHIVE)
  execute_process(COMMAND ${CMAKE_AR} rcs
    ${RESOURCE_DIRECTORY_ARCHIVE} ${resource_directory_object_files}
    COMMAND_ERROR_IS_FATAL ANY)
endif()
