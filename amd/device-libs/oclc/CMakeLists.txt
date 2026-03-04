##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Explicit mapping: TARGET_NAME|ISA_VERSION (using | as separator to avoid CMake list issues)
set(OCLC_ISA_TARGETS
  # GFX6
  "600|6000"
  "601|6001"
  "602|6002"
  # GFX7
  "700|7000"
  "701|7001"
  "702|7002"
  "703|7003"
  "704|7004"
  "705|7005"
  # GFX8
  "801|8001"
  "802|8002"
  "803|8003"
  "805|8005"
  "810|8100"
  # GFX9
  "900|9000"
  "902|9002"
  "904|9004"
  "906|9006"
  "908|9008"
  "909|9009"
  "90a|9010"
  "90c|9012"
  "942|9402"
  "950|9500"
  "9-generic|9000"
  "9-4-generic|9402"
  # GFX10
  "1010|10100"
  "1011|10101"
  "1012|10102"
  "1013|10103"
  "1030|10300"
  "1031|10301"
  "1032|10302"
  "1033|10303"
  "1034|10304"
  "1035|10305"
  "1036|10306"
  "10-1-generic|10100"
  "10-3-generic|10300"
  # GFX11
  "1100|11000"
  "1101|11001"
  "1102|11002"
  "1103|11003"
  "1150|11500"
  "1151|11501"
  "1152|11502"
  "1153|11503"
  "11-generic|11003"
  # GFX12
  "1200|12000"
  "1201|12001"
  "1250|12500"
  "1251|12501"
  "12-generic|12000"
)

# ABI versions (version in name equals value)
set(OCLC_ABI_VERSIONS 400 500 600)

# Generate ISA version files
foreach(entry ${OCLC_ISA_TARGETS})
  string(REPLACE "|" ";" entry_list "${entry}")
  list(GET entry_list 0 TARGET_NAME)
  list(GET entry_list 1 ISA_VERSION)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/isa_version.cl.in
    ${CMAKE_CURRENT_BINARY_DIR}/isa_version_${TARGET_NAME}.cl
    @ONLY
  )

  opencl_bc_lib(NAME oclc_isa_version_${TARGET_NAME}
                SOURCES ${CMAKE_CURRENT_BINARY_DIR}/isa_version_${TARGET_NAME}.cl)
endforeach()

# Generate ABI version files
foreach(ABI_VERSION ${OCLC_ABI_VERSIONS})
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/abi_version.cl.in
    ${CMAKE_CURRENT_BINARY_DIR}/abi_version_${ABI_VERSION}.cl
    @ONLY
  )

  opencl_bc_lib(NAME oclc_abi_version_${ABI_VERSION}
                SOURCES ${CMAKE_CURRENT_BINARY_DIR}/abi_version_${ABI_VERSION}.cl)
endforeach()

# Process remaining source files (excluding generated isa_version and abi_version)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl)
foreach(file ${sources})
  get_filename_component(name ${file} NAME_WE)
  get_filename_component(ext ${file} EXT)
  opencl_bc_lib(NAME oclc_${name} SOURCES ${file})
endforeach()
