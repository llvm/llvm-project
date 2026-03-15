set(INC_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)


set(RUNTIME_TARGET_DEPENDENCIES)

foreach(runtime ${LLVM_ENABLE_RUNTIMES})
  # FIXME: Some runtimes don't define a top level target that matches
  # the project name
  list(APPEND RUNTIME_TARGET_DEPENDENCIES $<TARGET_NAME_IF_EXISTS:${runtime}>)
endforeach()

set(GEN_RESOURCE_DIR_FILE ${LIB_DIR}/resource_dir.cpp)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CLANG_RESOURCE_DIR "lib/clang/${LLVM_VERSION_MAJOR}")
else()
  # TODO: This should be the only supported build path
  include(GetClangResourceDir)
  get_clang_resource_dir(CLANG_RESOURCE_DIR PREFIX ${LLVM_BINARY_DIR})
endif()

if(COMGR_USE_INCBIN)
  set(resource_directory_object_archive ${LIB_DIR}/resource_directory.a)
  set(tool_depends ${CMAKE_AR})
else()
  # FIXME: Dependency hack. This is an output file which is never
  # produced. This creates a file dependency between the
  # add_custom_command and the embed-resource-dir target. When using
  # #embed or bc2h, we are generating a c++ source added to a library
  # target. For some reason we need an additional dependency not added
  # to a build target in order to ensure embed-resource-dir is rebuilt
  # on resource directory content changes.
  set(non_source_dependency artificial_non_source_dependency)
endif()

# Detect the files that will be embedded from the built resource
# directory, so that there is a content dependency.
#
# TODO: It would be better if the runtimes build exported specific
# targets in a structured way instead of adding direct file
# dependencies.
#
# Keep this in sync with EmbedResourceDir.cmake
file(GLOB_RECURSE embedded_files
     LIST_DIRECTORIES false
     CONFIGURE_DEPENDS
     "${CLANG_RESOURCE_DIR}/lib/amd*/*.bc"
     "${CLANG_RESOURCE_DIR}/lib/amd*/*.a")

# TODO: Stop using bc2h. Really we ought to be able to rely on #embed,
# but it's not supported by the oldest supported versions of host
# compilers. Until then, this should switch to rc on windows to embed
# the binaries.
#
# TODO: Also compress this
add_custom_command(
  OUTPUT ${GEN_RESOURCE_DIR_FILE} ${resource_directory_object_archive} ${non_source_dependency}
  COMMAND ${CMAKE_COMMAND}
    -DBC2H_BINARY=$<TARGET_FILE:bc2h>
    -DGEN_RESOURCE_DIR_FILE=${GEN_RESOURCE_DIR_FILE}
    -DCLANG_RESOURCE_DIR=${CLANG_RESOURCE_DIR}
    -DCOMGR_USE_EMBED=${COMGR_USE_EMBED}
    -DCOMGR_USE_INCBIN=${COMGR_USE_INCBIN}
    -DOBJCOPY_OUTPUT_FORMAT=${OBJCOPY_OUTPUT_FORMAT}
    -DCMAKE_AR=${CMAKE_AR}
    -DCMAKE_OBJCOPY=${CMAKE_OBJCOPY}
    -DCMAKE_ASM_COMPILER=${CMAKE_ASM_COMPILER}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedResourceDir.cmake
  DEPENDS bc2h
          ${LLVM_ENABLE_RUNTIMES}
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedResourceDir.cmake
          ${RESOURCE_DIRECTORY_DEPENDENCIES}
          ${tool_depends}
          ${embedded_files}
  COMMENT "Embedding clang resource directory"
  WORKING_DIRECTORY ${LIB_DIR}
  USES_TERMINAL
  VERBATIM)

add_custom_target(embed-resource-dir DEPENDS
     ${GEN_RESOURCE_DIR_FILE}
     ${resource_directory_object_archive}
     ${non_source_dependency})

# This must not directly add GEN_RESOURCE_DIR_FILE as a source file of
# the library here. This must create the library, add the dependency
# on the custom target before adding the source to the library target.
add_library(embed-resource-dir-lib OBJECT)
set_target_properties(embed-resource-dir-lib PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED Yes
  CXX_EXTENSIONS No)
add_dependencies(embed-resource-dir-lib embed-resource-dir)
target_sources(embed-resource-dir-lib PRIVATE ${GEN_RESOURCE_DIR_FILE})

target_include_directories(embed-resource-dir-lib PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(embed-resource-dir-lib PRIVATE ${LLVM_LIBS})

if(resource_directory_object_archive)
  target_link_libraries(amd_comgr PRIVATE ${resource_directory_object_archive})
endif()

target_include_directories(embed-resource-dir-lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(amd_comgr PRIVATE embed-resource-dir-lib)


set(GEN_LIBRARY_INC_FILE ${INC_DIR}/libraries.inc)
set(GEN_LIBRARY_DEFS_INC_FILE ${INC_DIR}/libraries_defs.inc)

# cmake does not provide a way to query targets produced by a project,
# so we have to make one up. Ordinarily, individual library target
# names are usable. In this case, we don't want to have to maintain a
# list of bitcode libraries, since they change (e.g. when a new
# subtarget specific device library is added)
#
# If we found the device libraries through find_package, we were
# already provided a list of targets. If not, we tracked this in a
# global property. This is the same technique used for LLVM_LIBS in
# AddLLVM.

if(NOT DEFINED AMD_DEVICE_LIBS_TARGETS)
  get_property(AMD_DEVICE_LIBS_TARGETS GLOBAL PROPERTY AMD_DEVICE_LIBS)
endif()

if(NOT AMD_DEVICE_LIBS_TARGETS)
  message(FATAL_ERROR "Could not find list of device libraries")
endif()

set(TARGETS_INCLUDES "")
foreach(AMDGCN_LIB_TARGET ${AMD_DEVICE_LIBS_TARGETS})
  set(header ${AMDGCN_LIB_TARGET}.inc)

  # FIXME: It's very awkward to deal with the device library
  # build. Really, they are custom targets that do not nicely fit into
  # any of cmake's library concepts. However, they are artificially
  # exported as static libraries. The custom target has the
  # OUTPUT_NAME property, but imported libraries have the LOCATION
  # property.
  get_target_property(bc_lib_path ${AMDGCN_LIB_TARGET} LOCATION)
  if(NOT bc_lib_path)
    get_target_property(bc_lib_path ${AMDGCN_LIB_TARGET} OUTPUT_NAME)
  endif()

  if(NOT bc_lib_path)
    message(FATAL_ERROR "Could not find path to bitcode library")
  endif()

  # Generic targets contain - in the name, but that's not a valid C++
  # identifier so we need to replace - with _.
  string(REPLACE "-" "_" AMDGCN_LIB_TARGET_ID ${AMDGCN_LIB_TARGET})

  add_custom_command(OUTPUT ${INC_DIR}/${header}
    COMMAND bc2h ${bc_lib_path}
                 ${INC_DIR}/${header}
                 "${AMDGCN_LIB_TARGET_ID}_lib"
    DEPENDS bc2h ${AMDGCN_LIB_TARGET} ${bc_lib_path} ${bc_lib_path}
    COMMENT "Generating ${AMDGCN_LIB_TARGET}.inc"
  )
  set_property(DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES ${INC_DIR}/${header})

  add_custom_target(${AMDGCN_LIB_TARGET}_header DEPENDS ${INC_DIR}/${header})
  add_dependencies(amd_comgr ${AMDGCN_LIB_TARGET}_header)

  list(APPEND TARGETS_INCLUDES "#include \"${header}\"")
  list(APPEND TARGETS_HEADERS_FILENAME "${header}")
  list(APPEND TARGETS_HEADERS_REALPATH "${INC_DIR}/${header}")
endforeach()

list(JOIN TARGETS_INCLUDES "\n" TARGETS_INCLUDES)
file(GENERATE OUTPUT ${GEN_LIBRARY_INC_FILE} CONTENT "${TARGETS_INCLUDES}")

add_custom_command(OUTPUT ${INC_DIR}/opencl-c-base.inc
  COMMAND bc2h ${OPENCL_C_H}
                ${INC_DIR}/opencl-c-base.inc
                opencl_c_base
  DEPENDS bc2h clang ${OPENCL_C_H}
  COMMENT "Generating opencl-c-base.inc"
)
set_property(DIRECTORY APPEND PROPERTY
  ADDITIONAL_MAKE_CLEAN_FILES ${INC_DIR}/opencl-c-base.inc)
add_custom_target(opencl-c-base.inc_target DEPENDS ${INC_DIR}/opencl-c-base.inc)
add_dependencies(amd_comgr opencl-c-base.inc_target)

set(TARGETS_DEFS "")
list(APPEND TARGETS_DEFS "#ifndef AMD_DEVICE_LIBS_TARGET\n#define AMD_DEVICE_LIBS_TARGET(t)\n#endif")
list(APPEND TARGETS_DEFS "#ifndef AMD_DEVICE_LIBS_GFXIP\n#define AMD_DEVICE_LIBS_GFXIP(t, g)\n#endif")
list(APPEND TARGETS_DEFS "#ifndef AMD_DEVICE_LIBS_FUNCTION\n#define AMD_DEVICE_LIBS_FUNCTION(t, f)\n#endif")
list(APPEND TARGETS_DEFS "")
foreach(AMDGCN_LIB_TARGET ${AMD_DEVICE_LIBS_TARGETS})
  # Generic targets contain - in the name, but that's not a valid C++
  # identifier so we need to replace - with _.
  string(REPLACE "-" "_" AMDGCN_LIB_TARGET_ID ${AMDGCN_LIB_TARGET})

  list(APPEND TARGETS_DEFS "AMD_DEVICE_LIBS_TARGET(${AMDGCN_LIB_TARGET_ID})")
  # Generate function to select libraries for a given GFXIP number.
  if (${AMDGCN_LIB_TARGET} MATCHES "^oclc_isa_version_.+$")
    string(REGEX REPLACE "^oclc_isa_version_(.+)$" "\\1" gfxip ${AMDGCN_LIB_TARGET})
    list(APPEND TARGETS_DEFS "AMD_DEVICE_LIBS_GFXIP(${AMDGCN_LIB_TARGET_ID}, \"${gfxip}\")")
  endif()
  # Generate function to select libraries for given feature.
  if (${AMDGCN_LIB_TARGET} MATCHES "^oclc_.*_on$")
    string(REGEX REPLACE "^oclc_(.*)_on" "\\1" function ${AMDGCN_LIB_TARGET})
    list(APPEND TARGETS_DEFS "AMD_DEVICE_LIBS_FUNCTION(${AMDGCN_LIB_TARGET}, ${function})")
  endif()
endforeach()

list(APPEND TARGETS_DEFS "")
list(APPEND TARGETS_DEFS "#undef AMD_DEVICE_LIBS_TARGET")
list(APPEND TARGETS_DEFS "#undef AMD_DEVICE_LIBS_GFXIP")
list(APPEND TARGETS_DEFS "#undef AMD_DEVICE_LIBS_FUNCTION")

list(JOIN TARGETS_DEFS "\n" TARGETS_DEFS)
file(GENERATE OUTPUT ${GEN_LIBRARY_DEFS_INC_FILE} CONTENT "${TARGETS_DEFS}")

# compute the sha256 of the device libraries to detect changes and pass them to comgr (used by the cache)
find_package(Python3 REQUIRED Interpreter)
set(DEVICE_LIBS_ID_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/device-libs-id.py")
set(DEVICE_LIBS_ID_HEADER ${INC_DIR}/libraries_sha.inc)
add_custom_command(OUTPUT ${DEVICE_LIBS_ID_HEADER}
    COMMAND ${Python3_EXECUTABLE} ${DEVICE_LIBS_ID_SCRIPT} --varname DEVICE_LIBS_ID --output ${DEVICE_LIBS_ID_HEADER} --parent-directory ${INC_DIR} ${TARGETS_HEADERS_FILENAME}
    DEPENDS ${DEVICE_LIBS_ID_SCRIPT} ${TARGETS_HEADERS_REALPATH}
    COMMENT "Generating ${INC_DIR}/libraries_sha.inc"
)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${INC_DIR}/libraries_sha.inc)
add_custom_target(libraries_sha_header DEPENDS ${INC_DIR}/libraries_sha.inc)
add_dependencies(amd_comgr libraries_sha_header)

include_directories(${INC_DIR})
