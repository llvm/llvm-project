set(INC_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

file(WRITE ${INC_DIR}/libraries.inc "// Automatically generated file; DO NOT EDIT.\n")

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

  add_custom_command(OUTPUT ${INC_DIR}/${header}
    COMMAND bc2h ${bc_lib_path}
                 ${INC_DIR}/${header}
                 "${AMDGCN_LIB_TARGET}_lib"
    DEPENDS bc2h ${AMDGCN_LIB_TARGET} ${bc_lib_path}
    COMMENT "Generating ${AMDGCN_LIB_TARGET}.inc"
  )
  set_property(DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES ${INC_DIR}/${header})

  add_custom_target(${AMDGCN_LIB_TARGET}_header DEPENDS ${INC_DIR}/${header})
  add_dependencies(amd_comgr ${AMDGCN_LIB_TARGET}_header)

  file(APPEND ${INC_DIR}/libraries.inc "#include \"${header}\"\n")
endforeach()

add_custom_command(OUTPUT ${INC_DIR}/opencl1.2-c.inc
  COMMAND bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl1.2-c.pch
               ${INC_DIR}/opencl1.2-c.inc
               opencl1_2_c
  DEPENDS bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl1.2-c.pch
  COMMENT "Generating opencl1.2-c.inc"
)
set_property(DIRECTORY APPEND PROPERTY
  ADDITIONAL_MAKE_CLEAN_FILES ${INC_DIR}/opencl1.2-c.inc)
add_custom_target(opencl1.2-c.inc_target DEPENDS ${INC_DIR}/opencl1.2-c.inc)
add_dependencies(amd_comgr opencl1.2-c.inc_target)
file(APPEND ${INC_DIR}/libraries.inc "#include \"opencl1.2-c.inc\"\n")

add_custom_command(OUTPUT ${INC_DIR}/opencl2.0-c.inc
  COMMAND bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.pch
               ${INC_DIR}/opencl2.0-c.inc
               opencl2_0_c
  DEPENDS bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.pch
  COMMENT "Generating opencl2.0-c.inc"
)
set_property(DIRECTORY APPEND PROPERTY
  ADDITIONAL_MAKE_CLEAN_FILES ${INC_DIR}/opencl2.0-c.inc)
add_custom_target(opencl2.0-c.inc_target DEPENDS ${INC_DIR}/opencl2.0-c.inc)
add_dependencies(amd_comgr opencl2.0-c.inc_target)
file(APPEND ${INC_DIR}/libraries.inc "#include \"opencl2.0-c.inc\"\n")

# Generate function to select libraries for a given GFXIP number.
file(APPEND ${INC_DIR}/libraries.inc "#include \"llvm/ADT/StringRef.h\"\n")
file(APPEND ${INC_DIR}/libraries.inc
  "static std::tuple<const char*, const void*, size_t> get_oclc_isa_version(llvm::StringRef gfxip) {")
foreach(AMDGCN_LIB_TARGET ${AMD_DEVICE_LIBS_TARGETS})
  if (${AMDGCN_LIB_TARGET} MATCHES "^oclc_isa_version_.+$")
    string(REGEX REPLACE "^oclc_isa_version_(.+)$" "\\1" gfxip ${AMDGCN_LIB_TARGET})
    file(APPEND ${INC_DIR}/libraries.inc
      "if (gfxip == \"${gfxip}\") return std::make_tuple(\"${AMDGCN_LIB_TARGET}.bc\", ${AMDGCN_LIB_TARGET}_lib, ${AMDGCN_LIB_TARGET}_lib_size);")
  endif()
endforeach()
file(APPEND ${INC_DIR}/libraries.inc
  "return std::make_tuple(nullptr, nullptr, 0); }")

# Generate function to select libraries for given feature.
foreach(AMDGCN_LIB_TARGET ${AMD_DEVICE_LIBS_TARGETS})
  if (${AMDGCN_LIB_TARGET} MATCHES "^oclc_.*_on$")
    string(REGEX REPLACE "^oclc_(.*)_on" "\\1" function ${AMDGCN_LIB_TARGET})
    file(APPEND ${INC_DIR}/libraries.inc
      "static std::tuple<const char*, const void*, size_t> get_oclc_${function}(bool on) { \
       return std::make_tuple( \
         on ? \"oclc_${function}_on_lib.bc\" : \"oclc_${function}_off_lib.bc\", \
         on ? oclc_${function}_on_lib : oclc_${function}_off_lib, \
         on ? oclc_${function}_on_lib_size : oclc_${function}_off_lib_size \
       ); }")
  endif()
endforeach()

include_directories(${INC_DIR})
