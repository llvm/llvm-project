cmake_minimum_required(VERSION 3.20.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(libclc VERSION 0.2.0 LANGUAGES CXX C)
endif()
set(LLVM_SUBPROJECT_TITLE "libclc")

set(CMAKE_CXX_STANDARD 17)

# Add path for custom modules
list( INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules" )

enable_language( CLC )

include( AddLibclc )

include( GNUInstallDirs )

set( LIBCLC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )

option(
  LIBCLC_USE_SPIRV_BACKEND "Build SPIR-V targets with the SPIR-V backend." OFF
)

# List of all supported targets.
set( LIBCLC_TARGETS_ALL
  amdgcn-amd-amdhsa-llvm
  clspv--
  clspv64--
  nvptx64--
  nvptx64--nvidiacl
  nvptx64-nvidia-cuda
  spirv-mesa3d-
  spirv64-mesa3d-
)

set(LIBCLC_TARGET ${LLVM_DEFAULT_TARGET_TRIPLE})

if(NOT LIBCLC_TARGET)
  message(FATAL_ERROR "libclc target is empty\n")
endif()
if(NOT "${LIBCLC_TARGET}" IN_LIST LIBCLC_TARGETS_ALL)
  message(FATAL_ERROR "Unknown libclc target: ${LIBCLC_TARGET}\n"
    "Valid targets are: ${LIBCLC_TARGETS_ALL}\n")
endif()

if( LIBCLC_STANDALONE_BUILD OR CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  set( LIBCLC_STANDALONE_BUILD TRUE )

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_DIR}")
  include(AddLLVM)

  foreach( tool IN ITEMS llvm-link opt )
    find_program( LLVM_TOOL_${tool} ${tool} PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH )
    set( ${tool}_exe ${LLVM_TOOL_${tool}} )
    set( ${tool}_target )
  endforeach()

  # Setup the paths where libclc runtimes should be stored.
  set( LIBCLC_OUTPUT_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR} )
  set( LIBCLC_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/clc )
else()
  set( LIBCLC_STANDALONE_BUILD FALSE )

  # Note that we check this later (for both build types) but we can provide a
  # more useful error message when built in-tree. We assume that LLVM tools are
  # always available so don't warn here.
  if( NOT LLVM_RUNTIMES_BUILD AND NOT clang IN_LIST LLVM_ENABLE_PROJECTS )
    message(FATAL_ERROR "Clang is not enabled, but is required to build libclc in-tree")
  endif()

  # The package version is required to find the resource directory.
  if( LLVM_PACKAGE_VERSION AND NOT PACKAGE_VERSION )
    set( PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}" )
  endif()

  get_host_tool_path( llvm-link LLVM_LINK llvm-link_exe llvm-link_target )
  get_host_tool_path( opt OPT opt_exe opt_target )

  # Setup the paths where libclc runtimes should be stored. By default, in an
  # in-tree build we place the libraries in clang's resource driectory.
  include(GetClangResourceDir)
  get_clang_resource_dir( LIBCLC_INSTALL_DIR )
  cmake_path( APPEND LIBCLC_INSTALL_DIR "lib" )

  cmake_path( GET LLVM_LIBRARY_OUTPUT_INTDIR PARENT_PATH LIBCLC_OUTPUT_LIBRARY_DIR )
  cmake_path( APPEND LIBCLC_OUTPUT_LIBRARY_DIR ${LIBCLC_INSTALL_DIR} )
endif()

if( NOT LIBCLC_USE_SPIRV_BACKEND )
  # llvm-spirv is an optional dependency, used to build spirv-* targets when
  # the SPIR-V backend hasn't been requested. It may be provided in-tree or
  # externally.
  if( TARGET llvm-spirv )
    get_host_tool_path( llvm-spirv LLVM_SPIRV llvm-spirv_exe llvm-spirv_target )
  else()
    find_program( LLVM_SPIRV llvm-spirv HINTS ${LLVM_TOOLS_BINARY_DIR} )
    set( llvm-spirv_exe "${LLVM_SPIRV}" )
  endif()
endif()

message(STATUS "libclc target '${LIBCLC_TARGET}' is enabled")

string( REPLACE "-" ";" TRIPLE  ${LIBCLC_TARGET} )
list(GET TRIPLE 0 ARCH)
list(GET TRIPLE 2 OS)

if(ARCH STREQUAL spirv OR ARCH STREQUAL spirv64)
  if(NOT LIBCLC_USE_SPIRV_BACKEND AND NOT llvm-spirv_exe)
    message(FATAL_ERROR "SPIR-V backend or llvm-spirv is required for libclc ${LIBCLC_TARGET}")
  endif()
endif()

foreach( tool IN ITEMS opt llvm-link )
  if( NOT EXISTS "${${tool}_exe}" AND "${${tool}_target}" STREQUAL "" )
    message( FATAL_ERROR "libclc toolchain incomplete - missing tool ${tool}!" )
  endif()
endforeach()

add_subdirectory(clc/lib/generic)
add_subdirectory(opencl/lib/generic)

if(ARCH STREQUAL amdgcn)
  add_subdirectory(clc/lib/amdgpu)
  add_subdirectory(opencl/lib/amdgpu)
elseif(ARCH STREQUAL nvptx64)
  add_subdirectory(clc/lib/ptx-nvidiacl)
elseif(ARCH STREQUAL spirv OR ARCH STREQUAL spirv64)
  add_subdirectory(clc/lib/spirv)
  add_subdirectory(opencl/lib/spirv)
elseif(ARCH STREQUAL clspv OR ARCH STREQUAL clspv64)
  add_subdirectory(clc/lib/clspv)
  add_subdirectory(opencl/lib/clspv)
endif()

add_custom_target( libclc ALL )

add_custom_target( libclc-opencl-builtins COMMENT "Build libclc OpenCL builtins" )
add_dependencies( libclc libclc-opencl-builtins )

# Determine the clang target triple.
set(clang_triple ${LIBCLC_TARGET})
if(ARCH STREQUAL spirv AND LIBCLC_USE_SPIRV_BACKEND)
  set(clang_triple spirv32--)
elseif(ARCH STREQUAL spirv64 AND LIBCLC_USE_SPIRV_BACKEND)
  set(clang_triple spirv64--)
elseif(ARCH STREQUAL spirv OR ARCH STREQUAL clspv)
  set(clang_triple spir--)
elseif(ARCH STREQUAL spirv64 OR ARCH STREQUAL clspv64)
  set(clang_triple spir64--)
endif()

# Determine the preprocessor identifier for this target.
set(MACRO_ARCH ${ARCH})
if(ARCH STREQUAL spirv)
  set(MACRO_ARCH SPIRV32)
elseif(ARCH STREQUAL spirv64)
  set(MACRO_ARCH SPIRV64)
elseif(ARCH STREQUAL clspv)
  set(MACRO_ARCH CLSPV32)
elseif(ARCH STREQUAL clspv64)
  set(MACRO_ARCH CLSPV64)
endif()
string(TOUPPER "CLC_${MACRO_ARCH}" target_define)

# Address space values.
set(private_addrspace_val 0)
set(generic_addrspace_val 0)
if(ARCH STREQUAL amdgcn)
  set(private_addrspace_val 5)
endif()
if(ARCH STREQUAL spirv OR ARCH STREQUAL spirv64)
  set(generic_addrspace_val 4)
endif()

# Target-specific compile flags and defines.
set(target_compile_flags)
set(target_extra_defines)
set(opt_flags -O3)

if(ARCH STREQUAL spirv OR ARCH STREQUAL spirv64)
  list(APPEND target_compile_flags -O0 -finline-hint-functions)
  list(APPEND target_extra_defines CLC_SPIRV)
  set(opt_flags)
elseif(ARCH STREQUAL clspv OR ARCH STREQUAL clspv64)
  list(APPEND target_compile_flags -Wno-unknown-assumption -U__opencl_c_int64)
  list(APPEND target_extra_defines CLC_CLSPV)
elseif(ARCH STREQUAL amdgcn)
  list(APPEND target_compile_flags "SHELL:-Xclang -mcode-object-version=none")
endif()

# Collect CLC sources; target-specific sources override generic ones by basename.
set(_clc_overrides)
if(ARCH STREQUAL amdgcn)
  list(APPEND _clc_overrides ${CLC_AMDGPU_SOURCES})
elseif(ARCH STREQUAL nvptx64 AND (OS STREQUAL nvidiacl OR OS STREQUAL cuda))
  list(APPEND _clc_overrides ${CLC_PTX_NVIDIACL_SOURCES})
elseif(ARCH STREQUAL spirv OR ARCH STREQUAL spirv64)
  list(APPEND _clc_overrides ${CLC_SPIRV_SOURCES})
elseif(ARCH STREQUAL clspv OR ARCH STREQUAL clspv64)
  list(APPEND _clc_overrides ${CLC_CLSPV_SOURCES})
endif()
libclc_merge_sources(clc_sources ${CLC_GENERIC_SOURCES} ${_clc_overrides})

# Collect OpenCL sources. SPIR-V and Clspv targets use self-contained
# subsets while others merge with target-specific overrides.
if(ARCH STREQUAL spirv OR ARCH STREQUAL spirv64)
  set(opencl_sources ${OPENCL_SPIRV_SOURCES})
elseif(ARCH STREQUAL clspv OR ARCH STREQUAL clspv64)
  set(opencl_sources ${OPENCL_CLSPV_SOURCES})
else()
  set(_opencl_overrides)
  if(ARCH STREQUAL amdgcn)
    list(APPEND _opencl_overrides ${OPENCL_AMDGCN_SOURCES})
  endif()
  libclc_merge_sources(opencl_sources
    ${OPENCL_GENERIC_SOURCES} ${_opencl_overrides})
endif()

# Common compile options shared by CLC and OpenCL libraries.
set(compile_flags
  -flto
  --target=${clang_triple}
  -nostdlib
  -nostdlibinc
  -cl-no-stdinc
  -cl-std=CL3.0
  -include opencl-c-base.h
  -Werror=undef
  -Wall
  -Wextra
  -fdiscard-value-names
  -ffp-contract=fast-honor-pragmas
  -fdenormal-fp-math=dynamic
  ${target_compile_flags}
)

set(_common_defs
  ${target_define}
  ${target_extra_defines}
  __CLC_PRIVATE_ADDRSPACE_VAL=${private_addrspace_val}
  __CLC_GENERIC_ADDRSPACE_VAL=${generic_addrspace_val}
)

# Build the CLC internal builtins library.
string(REPLACE "-" "_" lib_suffix ${LIBCLC_TARGET})
set(clc_lib clc_builtins_${lib_suffix})
add_libclc_builtin_library(${clc_lib}
  SOURCES ${clc_sources}
  COMPILE_OPTIONS ${compile_flags}
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/clc/include
  COMPILE_DEFINITIONS ${_common_defs}
  FOLDER "libclc/Device IR/CLC"
)

# Build, link, and install the final OpenCL builtins library.
add_libclc_library(libclc-${LIBCLC_TARGET}
  ARCH ${ARCH}
  TRIPLE ${clang_triple}
  TARGET_TRIPLE ${LIBCLC_TARGET}
  SOURCES ${opencl_sources}
  COMPILE_OPTIONS ${compile_flags} "SHELL:-Xclang -fdeclare-opencl-builtins"
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/clc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencl/include
  COMPILE_DEFINITIONS ${_common_defs}
  INTERNALIZE_LIBRARIES ${clc_lib}
  OPT_FLAGS ${opt_flags}
  OUTPUT_FILENAME libclc
  PARENT_TARGET libclc-opencl-builtins
)

set(LIBCLC_UNRESOLVED_SYMBOL_TEST_TARGETS libclc-${LIBCLC_TARGET})

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
