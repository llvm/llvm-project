# LibcxxHeaders.cmake - Discover and embed libc++ headers for HIPRTC.
#
# Build process:
#   1. trace_headers.py runs clang -E -H → manifest TSV
#   2. EmbedLibcxxHeaders.cmake embeds via shared EmbedFiles.cmake → .cpp

set(LIBCXX_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib/libcxx)
file(MAKE_DIRECTORY ${LIBCXX_HEADERS_DIR})

set(COMGR_DEVICE_TRIPLE "amdgcn-amd-amdhsa")

# Top-level C++ headers to embed (freestanding-safe, no system C deps)
set(LIBCXX_USER_HEADERS
  type_traits
  limits
  tuple
  cstdint
  cstddef
  initializer_list
  concepts
)

# Find libc++ headers from the LLVM source tree
set(LIBCXX_SEARCH_PATHS
  "${CMAKE_SOURCE_DIR}/libcxx/include"
  "${CMAKE_SOURCE_DIR}/../libcxx/include"
  "${CMAKE_SOURCE_DIR}/../../libcxx/include"  # amd/comgr -> libcxx
)

set(LIBCXX_INCLUDE_DIR "")
foreach(PATH ${LIBCXX_SEARCH_PATHS})
  if(EXISTS "${PATH}")
    get_filename_component(LIBCXX_INCLUDE_DIR "${PATH}" ABSOLUTE)
    break()
  endif()
endforeach()

if(NOT LIBCXX_INCLUDE_DIR)
  message(STATUS "libc++ headers not found in: ${LIBCXX_SEARCH_PATHS}")
  message(STATUS "HIPRTC std header support will be disabled")
  return()
endif()

if(NOT TARGET clang)
  message(WARNING "clang target not found — cannot auto-discover headers. "
    "HIPRTC std header support will be disabled. "
    "Ensure find_package(Clang) is called before including LibcxxHeaders.")
  return()
endif()
set(CLANG_FOR_TRACE "$<TARGET_FILE:clang>")
set(CLANG_TRACE_DEPS clang)
if(TARGET clang-resource-headers)
  list(APPEND CLANG_TRACE_DEPS clang-resource-headers)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "Embedding libc++ headers from: ${LIBCXX_INCLUDE_DIR}")

set(HIPRTC_CONFIG_SITE "${CMAKE_CURRENT_SOURCE_DIR}/include/__config_site_hiprtc")
set(TRACE_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/trace_headers.py")
set(MANIFEST_FILE "${LIBCXX_HEADERS_DIR}/libcxx_manifest.tsv")
set(GEN_LIBCXX_FILE "${LIBCXX_HEADERS_DIR}/libcxx_headers.cpp")

if(COMGR_USE_INCBIN)
  set(libcxx_object_archive ${LIBCXX_HEADERS_DIR}/libcxx_headers.a)
  set(libcxx_tool_depends ${CMAKE_AR})
else()
  set(libcxx_non_source_dependency libcxx_artificial_dependency)
endif()

# Step 1: Trace headers at build time → manifest
add_custom_command(
  OUTPUT ${MANIFEST_FILE}
  COMMAND ${Python3_EXECUTABLE} ${TRACE_SCRIPT}
    --clang ${CLANG_FOR_TRACE}
    --libcxx-dir ${LIBCXX_INCLUDE_DIR}
    --config-site ${HIPRTC_CONFIG_SITE}
    --target ${COMGR_DEVICE_TRIPLE}
    --headers ${LIBCXX_USER_HEADERS}
    --output ${MANIFEST_FILE}
  DEPENDS
    ${CLANG_TRACE_DEPS}
    ${TRACE_SCRIPT}
    ${HIPRTC_CONFIG_SITE}
  COMMENT "Tracing libc++ header dependencies for HIPRTC"
  VERBATIM
)

# Step 2: Embed headers at build time → C++ source
add_custom_command(
  OUTPUT ${GEN_LIBCXX_FILE} ${libcxx_object_archive} ${libcxx_non_source_dependency}
  COMMAND ${CMAKE_COMMAND}
    -DBC2H_BINARY=$<TARGET_FILE:bc2h>
    -DGEN_LIBCXX_HEADERS_FILE=${GEN_LIBCXX_FILE}
    -DLIBCXX_MANIFEST_FILE=${MANIFEST_FILE}
    -DCOMGR_USE_EMBED=${COMGR_USE_EMBED}
    -DCOMGR_USE_INCBIN=${COMGR_USE_INCBIN}
    -DCMAKE_AR=${CMAKE_AR}
    -DCMAKE_ASM_COMPILER=${CMAKE_ASM_COMPILER}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedLibcxxHeaders.cmake
  DEPENDS
    bc2h
    ${MANIFEST_FILE}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedLibcxxHeaders.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFiles.cmake
    ${libcxx_tool_depends}
  COMMENT "Embedding libc++ headers for HIPRTC"
  WORKING_DIRECTORY ${LIBCXX_HEADERS_DIR}
  USES_TERMINAL
  VERBATIM
)

add_custom_target(embed-libcxx-headers DEPENDS
  ${GEN_LIBCXX_FILE}
  ${libcxx_object_archive}
  ${libcxx_non_source_dependency})

# Object library for the generated C++ source (mirrors embed-resource-dir-lib)
add_library(embed-libcxx-headers-lib OBJECT)
set_target_properties(embed-libcxx-headers-lib PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED Yes
  CXX_EXTENSIONS No
  POSITION_INDEPENDENT_CODE ON)
add_dependencies(embed-libcxx-headers-lib embed-libcxx-headers)
target_sources(embed-libcxx-headers-lib PRIVATE ${GEN_LIBCXX_FILE})

target_include_directories(embed-libcxx-headers-lib PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(libcxx_object_archive)
  target_link_libraries(amd_comgr PRIVATE ${libcxx_object_archive})
endif()

target_link_libraries(amd_comgr PRIVATE embed-libcxx-headers-lib)

target_compile_definitions(amd_comgr PRIVATE COMGR_HAS_LIBCXX_HEADERS=1)

message(STATUS "HIPRTC std headers will use standard clang include paths at runtime")
