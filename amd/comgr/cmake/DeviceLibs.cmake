set(INC_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

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
    DEPENDS bc2h ${AMDGCN_LIB_TARGET} ${bc_lib_path}
    COMMENT "Generating ${AMDGCN_LIB_TARGET}.inc"
  )
  set_property(DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES ${INC_DIR}/${header})

  add_custom_target(${AMDGCN_LIB_TARGET}_header DEPENDS ${INC_DIR}/${header})
  add_dependencies(amd_comgr ${AMDGCN_LIB_TARGET}_header)

  list(APPEND TARGETS_INCLUDES "#include \"${header}\"")
  list(APPEND TARGETS_HEADERS "${INC_DIR}/${header}")
endforeach()

list(JOIN TARGETS_INCLUDES "\n" TARGETS_INCLUDES)
file(GENERATE OUTPUT ${GEN_LIBRARY_INC_FILE} CONTENT "${TARGETS_INCLUDES}")

foreach(OPENCL_VERSION 1.2 2.0)
  string(REPLACE . _ OPENCL_UNDERSCORE_VERSION ${OPENCL_VERSION})
  add_custom_command(OUTPUT ${INC_DIR}/opencl${OPENCL_VERSION}-c.inc
    COMMAND bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl${OPENCL_VERSION}-c.pch
                 ${INC_DIR}/opencl${OPENCL_VERSION}-c.inc
                 opencl${OPENCL_UNDERSCORE_VERSION}_c
    DEPENDS bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl${OPENCL_VERSION}-c.pch
    COMMENT "Generating opencl${OPENCL_VERSION}-c.inc"
  )
  set_property(DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES ${INC_DIR}/opencl${OPENCL_VERSION}-c.inc)
  add_custom_target(opencl${OPENCL_VERSION}-c.inc_target DEPENDS ${INC_DIR}/opencl${OPENCL_VERSION}-c.inc)
  add_dependencies(amd_comgr opencl${OPENCL_VERSION}-c.inc_target)
endforeach()

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
  COMMAND ${Python3_EXECUTABLE} ${DEVICE_LIBS_ID_SCRIPT} --varname DEVICE_LIBS_ID --output ${DEVICE_LIBS_ID_HEADER} ${TARGETS_HEADERS}
  DEPENDS ${DEVICE_LIBS_ID_SCRIPT} ${TARGETS_HEADERS}
    COMMENT "Generating ${INC_DIR}/libraries_sha.inc"
)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${INC_DIR}/libraries_sha.inc)
add_custom_target(libraries_sha_header DEPENDS ${INC_DIR}/libraries_sha.inc)
add_dependencies(amd_comgr libraries_sha_header)

include_directories(${INC_DIR})
