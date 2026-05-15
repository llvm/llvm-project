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

# List of all supported architectures.
set( LIBCLC_ARCHS_NVPTX nvptx64 )
set( LIBCLC_ARCHS_AMDGPU amdgpu amdgcn )
set( LIBCLC_ARCHS_SPIRV spirv spirv32 spirv64 )
list( APPEND LIBCLC_ARCHS_ALL ${LIBCLC_ARCHS_AMDGPU} ${LIBCLC_ARCHS_NVPTX} ${LIBCLC_ARCHS_SPIRV} )

set(LIBCLC_TARGET ${LLVM_DEFAULT_TARGET_TRIPLE})

if(NOT LIBCLC_TARGET)
  message(FATAL_ERROR "libclc target is empty\n")
endif()

# Parse the target triple into arch and OS components.
string(REPLACE "-" ";" triple_components ${LIBCLC_TARGET})
list(GET triple_components 0 LIBCLC_TARGET_ARCH)
list(GET triple_components 2 LIBCLC_TARGET_OS)

if(NOT "${LIBCLC_TARGET_ARCH}" IN_LIST LIBCLC_ARCHS_ALL)
  message(FATAL_ERROR "Unknown libclc target architecture: ${LIBCLC_TARGET_ARCH}\n"
    "Target was: ${LIBCLC_TARGET}\n"
    "Valid architectures are: ${LIBCLC_ARCHS_ALL}\n")
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

# Map the LLVM target architecture to the standard directory name.
if(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_AMDGPU)
  set(LIBCLC_ARCH_DIR amdgpu)
elseif(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_NVPTX)
  set(LIBCLC_ARCH_DIR nvptx)
elseif(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_SPIRV)
  set(LIBCLC_ARCH_DIR spirv)
endif()

if(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_SPIRV)
  if(NOT LIBCLC_TARGET_OS STREQUAL vulkan AND NOT LIBCLC_USE_SPIRV_BACKEND AND NOT llvm-spirv_exe)
    message(FATAL_ERROR "SPIR-V backend or llvm-spirv is required for libclc ${LIBCLC_TARGET}")
  endif()
endif()

foreach( tool IN ITEMS opt llvm-link )
  if( NOT EXISTS "${${tool}_exe}" AND "${${tool}_target}" STREQUAL "" )
    message( FATAL_ERROR "libclc toolchain incomplete - missing tool ${tool}!" )
  endif()
endforeach()

# Determine the clang target triple. Vulkan and SPIR-V backend targets use the
# triple directly; other SPIR-V targets fall back to the legacy SPIR target.
set(clang_triple ${LIBCLC_TARGET})
if(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_SPIRV)
  if(NOT LIBCLC_TARGET_OS STREQUAL vulkan AND NOT LIBCLC_USE_SPIRV_BACKEND)
    if(LIBCLC_TARGET_ARCH STREQUAL spirv)
      set(clang_triple spir--)
    else()
      set(clang_triple spir64--)
    endif()
  endif()
endif()

# Address space values.
set(private_addrspace_val 0)
set(generic_addrspace_val 0)
if(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_AMDGPU)
  set(private_addrspace_val 5)
endif()
if(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_SPIRV AND NOT LIBCLC_TARGET_OS STREQUAL vulkan)
  set(generic_addrspace_val 4)
endif()

# Target-specific compile flags and defines.
set(target_compile_flags)
set(target_extra_defines)
set(opt_flags -O3)

if(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_SPIRV)
  if(LIBCLC_TARGET_OS STREQUAL vulkan)
    list(APPEND target_compile_flags -Wno-unknown-assumption -U__opencl_c_int64)
  else()
    list(APPEND target_compile_flags -O0 -finline-hint-functions)
    list(APPEND target_extra_defines CLC_SPIRV)
    set(opt_flags)
  endif()
elseif(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_AMDGPU)
  list(APPEND target_compile_flags "SHELL:-Xclang -mcode-object-version=none")
endif()
if(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_AMDGPU OR LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_NVPTX)
  list(APPEND target_compile_flags -fvisibility=hidden)
  list(APPEND target_compile_flags "SHELL:-Xclang -fapply-global-visibility-to-externs")
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

set(common_defs
  ${target_extra_defines}
  __CLC_PRIVATE_ADDRSPACE_VAL=${private_addrspace_val}
  __CLC_GENERIC_ADDRSPACE_VAL=${generic_addrspace_val}
)

add_custom_target(libclc ALL)
add_custom_target(libclc-opencl-builtins COMMENT "Build libclc OpenCL builtins")
add_dependencies(libclc libclc-opencl-builtins)

# Configure the CLC internal builtins library.
set(LIBCLC_CLC_TARGET clc)
libclc_add_builtin_library(${LIBCLC_CLC_TARGET}
  COMPILE_OPTIONS ${compile_flags}
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/clc/include
  COMPILE_DEFINITIONS ${common_defs}
  SOURCE_SUB_DIR clc
  FOLDER "libclc/Device IR/CLC"
)

# Build, link, and install the final OpenCL builtins library.
set(LIBCLC_OPENCL_TARGET opencl)
libclc_add_library(libclc-${LIBCLC_TARGET}
  ARCH ${LIBCLC_TARGET_ARCH}
  TRIPLE ${clang_triple}
  TARGET_TRIPLE ${LIBCLC_TARGET}
  COMPILE_OPTIONS ${compile_flags} "SHELL:-Xclang -fdeclare-opencl-builtins"
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/clc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencl/include
  COMPILE_DEFINITIONS ${common_defs}
  INTERNALIZE_LIBRARIES ${LIBCLC_CLC_TARGET}
  OPT_FLAGS ${opt_flags}
  SOURCE_TARGET ${LIBCLC_OPENCL_TARGET}
  SOURCE_SUB_DIR opencl
  OUTPUT_FILENAME libclc
  PARENT_TARGET libclc-opencl-builtins
)

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
