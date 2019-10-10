set(INC_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

file(WRITE ${INC_DIR}/libraries.inc "// Automatically generated file; DO NOT EDIT.\n")

foreach(AMDGCN_LIB_TARGET ${AMD_DEVICE_LIBS_TARGETS})
  set(header ${AMDGCN_LIB_TARGET}.inc)
  add_custom_command(OUTPUT ${INC_DIR}/${header}
    COMMAND bc2h $<TARGET_FILE:${AMDGCN_LIB_TARGET}>
                 ${INC_DIR}/${header}
                 ${AMDGCN_LIB_TARGET}
    DEPENDS bc2h ${AMDGCN_LIB_TARGET}
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
  if (${AMDGCN_LIB_TARGET} MATCHES "^oclc_isa_version_.+_lib$")
    string(REGEX REPLACE "^oclc_isa_version_(.+)_lib$" "\\1" gfxip ${AMDGCN_LIB_TARGET})
    file(APPEND ${INC_DIR}/libraries.inc
      "if (gfxip == \"${gfxip}\") return std::make_tuple(\"${AMDGCN_LIB_TARGET}.bc\", ${AMDGCN_LIB_TARGET}, ${AMDGCN_LIB_TARGET}_size);")
  endif()
endforeach()
file(APPEND ${INC_DIR}/libraries.inc
  "return std::make_tuple(nullptr, nullptr, 0); }")

# Generate function to select libraries for given feature.
foreach(AMDGCN_LIB_TARGET ${AMD_DEVICE_LIBS_TARGETS})
  if (${AMDGCN_LIB_TARGET} MATCHES "^oclc_.*_on_lib$")
    string(REGEX REPLACE "^oclc_(.*)_on_lib" "\\1" function ${AMDGCN_LIB_TARGET})
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
