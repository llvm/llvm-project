include_guard(GLOBAL)

include(FetchContent)

cmake_policy(PUSH)

# Direct population is required to load IGA's helpers before adding GED.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# Intel Graphics Compiler v2.38.2.
set(INTER_IGC_COMMIT 3eef0f89d3a4fe2b443de595e23d7700a5d1491b)
FetchContent_Declare(inter_igc
  URL
    "https://github.com/intel/intel-graphics-compiler/archive/${INTER_IGC_COMMIT}.tar.gz"
  URL_HASH
    SHA256=06a5b9d739be2b7655d399b14c80f045f957e353a4c28ddaf08de322124dcb70
)

FetchContent_GetProperties(inter_igc)
if(NOT inter_igc_POPULATED)
  FetchContent_Populate(inter_igc)
endif()

# GED's subtree expects these helpers from its parent IGA build.
include("${inter_igc_SOURCE_DIR}/visa/iga/BuildFunctions.cmake")
add_subdirectory(
  "${inter_igc_SOURCE_DIR}/visa/iga/GEDLibrary/GED_external"
  "${inter_igc_BINARY_DIR}/ged"
  EXCLUDE_FROM_ALL
)

set(GED_BRANCH GED_external)
set(LINK_AS_STATIC_LIB FALSE)
add_subdirectory(
  "${inter_igc_SOURCE_DIR}/visa/iga/IGALibrary"
  "${inter_igc_BINARY_DIR}/iga"
  EXCLUDE_FROM_ALL
)
unset(GED_BRANCH)
unset(LINK_AS_STATIC_LIB)

if(MSVC)
  target_compile_options(GEDLibrary PRIVATE /w)
  target_compile_options(IGA_OLIB PRIVATE /w)
  target_compile_options(IGA_SLIB PRIVATE /w)
else()
  target_compile_options(GEDLibrary PRIVATE -w)
  target_compile_options(IGA_OLIB PRIVATE -w)
  target_compile_options(IGA_SLIB PRIVATE -w)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(inter_ged_platform ia32)
else()
  set(inter_ged_platform intel64)
endif()

set(INTER_GED_INCLUDE_DIRS
  "${inter_igc_SOURCE_DIR}/visa/iga/GEDLibrary/GED_external/Source"
  "${inter_igc_SOURCE_DIR}/visa/iga/GEDLibrary/GED_external/Source/common"
  "${inter_igc_SOURCE_DIR}/visa/iga/GEDLibrary/GED_external/Source/ged"
  "${inter_igc_SOURCE_DIR}/visa/iga/GEDLibrary/GED_external/build/autogen-${inter_ged_platform}"
)

add_library(InterGED STATIC $<TARGET_OBJECTS:GEDLibrary>)
target_include_directories(InterGED SYSTEM PUBLIC ${INTER_GED_INCLUDE_DIRS})

target_include_directories(IGA_SLIB SYSTEM PUBLIC
  "${inter_igc_SOURCE_DIR}/visa/iga/IGALibrary/api")

cmake_policy(POP)
