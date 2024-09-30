cmake_minimum_required(VERSION 3.20.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(libclc VERSION 0.2.0 LANGUAGES CXX C)
endif()
set(LLVM_SUBPROJECT_TITLE "libclc")

set(CMAKE_CXX_STANDARD 17)

# Add path for custom modules
list( INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules" )

set( LIBCLC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set( LIBCLC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} )
set( LIBCLC_OBJFILE_DIR ${LIBCLC_BINARY_DIR}/obj.libclc.dir )

include( AddLibclc )

include( GNUInstallDirs )
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  amdgcn-amdhsa/lib/SOURCES;
  amdgcn/lib/SOURCES;
  amdgcn-mesa3d/lib/SOURCES;
  amdgpu/lib/SOURCES;
  clspv/lib/SOURCES;
  clspv64/lib/SOURCES;
  generic/lib/SOURCES;
  ptx/lib/SOURCES;
  ptx-nvidiacl/lib/SOURCES;
  r600/lib/SOURCES;
  spirv/lib/SOURCES;
  spirv64/lib/SOURCES
)

set( LIBCLC_MIN_LLVM 3.9.0 )

set( LIBCLC_TARGETS_TO_BUILD "all"
    CACHE STRING "Semicolon-separated list of libclc targets to build, or 'all'." )

option( ENABLE_RUNTIME_SUBNORMAL "Enable runtime linking of subnormal support." OFF )

if( LIBCLC_STANDALONE_BUILD OR CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  # Out-of-tree configuration
  set( LIBCLC_STANDALONE_BUILD TRUE )

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_DIR}")
  include(AddLLVM)

  message( STATUS "libclc LLVM version: ${LLVM_PACKAGE_VERSION}" )

  if( LLVM_PACKAGE_VERSION VERSION_LESS LIBCLC_MIN_LLVM )
    message( FATAL_ERROR "libclc needs at least LLVM ${LIBCLC_MIN_LLVM}" )
  endif()

  # Import required tools
  if( NOT EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )
    foreach( tool IN ITEMS clang llvm-as llvm-link opt )
      find_program( LLVM_TOOL_${tool} ${tool} PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH )
      set( ${tool}_exe ${LLVM_TOOL_${tool}} )
      set( ${tool}_target )
    endforeach()
  endif()
else()
  # In-tree configuration
  set( LIBCLC_STANDALONE_BUILD FALSE )

  set( LLVM_PACKAGE_VERSION ${LLVM_VERSION} )

  # Note that we check this later (for both build types) but we can provide a
  # more useful error message when built in-tree. We assume that LLVM tools are
  # always available so don't warn here.
  if( NOT clang IN_LIST LLVM_ENABLE_PROJECTS )
    message(FATAL_ERROR "Clang is not enabled, but is required to build libclc in-tree")
  endif()

  if( NOT EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )
    get_host_tool_path( clang CLANG clang_exe clang_target )
    get_host_tool_path( llvm-as LLVM_AS llvm-as_exe llvm-as_target )
    get_host_tool_path( llvm-link LLVM_LINK llvm-link_exe llvm-link_target )
    get_host_tool_path( opt OPT opt_exe opt_target )
  endif()
endif()

if( EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )
  message( WARNING "Using custom LLVM tools to build libclc: "
    "${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR}, "
    " ensure the tools are up to date." )
  # Note - use a differently named variable than LLVM_TOOL_${tool} as above, as
  # the variable name is used to cache the result of find_program. If we used
  # the same name, a user wouldn't be able to switch a build between default
  # and custom tools.
  foreach( tool IN ITEMS clang llvm-as llvm-link opt )
    find_program( LLVM_CUSTOM_TOOL_${tool} ${tool}
      PATHS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH )
    set( ${tool}_exe ${LLVM_CUSTOM_TOOL_${tool}} )
    set( ${tool}_target )
  endforeach()
endif()

foreach( tool IN ITEMS clang opt llvm-as llvm-link )
  if( NOT EXISTS "${${tool}_exe}" AND "${tool}_target" STREQUAL "" )
    message( FATAL_ERROR "libclc toolchain incomplete - missing tool ${tool}!" )
  endif()
endforeach()

# llvm-spirv is an optional dependency, used to build spirv-* targets.
# It may be provided in-tree or externally.
if( TARGET llvm-spirv )
  get_host_tool_path( llvm-spirv LLVM_SPIRV llvm-spirv_exe llvm-spirv_target )
else()
  find_program( LLVM_SPIRV llvm-spirv PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH )
  set( llvm-spirv_exe "${LLVM_SPIRV}" )
  set( llvm-spirv_target )
endif()

# List of all targets. Note that some are added dynamically below.
set( LIBCLC_TARGETS_ALL
  amdgcn--
  amdgcn--amdhsa
  clspv--
  clspv64--
  r600--
  nvptx--
  nvptx64--
  nvptx--nvidiacl
  nvptx64--nvidiacl
)

# mesa3d environment is only available since LLVM 4.0
if( LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL 4.0.0 )
  list( APPEND LIBCLC_TARGETS_ALL amdgcn-mesa-mesa3d )
endif()

# spirv-mesa3d and spirv64-mesa3d targets can only be built with the (optional)
# llvm-spirv external tool.
if( llvm-spirv_exe )
  list( APPEND LIBCLC_TARGETS_ALL  spirv-mesa3d- spirv64-mesa3d- )
endif()

if( LIBCLC_TARGETS_TO_BUILD STREQUAL "all" )
  set( LIBCLC_TARGETS_TO_BUILD ${LIBCLC_TARGETS_ALL} )
endif()

list( SORT LIBCLC_TARGETS_TO_BUILD )

# Verify that the user hasn't requested mesa3d targets without an available
# llvm-spirv tool.
if( "spirv-mesa3d-" IN_LIST LIBCLC_TARGETS_TO_BUILD OR "spirv64-mesa3d-" IN_LIST LIBCLC_TARGETS_TO_BUILD )
  if( NOT llvm-spirv_exe )
    message( FATAL_ERROR "SPIR-V targets requested, but spirv-tools is not installed" )
  endif()
endif()

# Construct LLVM version define
set( LLVM_VERSION_DEFINE "-DHAVE_LLVM=0x${LLVM_VERSION_MAJOR}0${LLVM_VERSION_MINOR}" )

# This needs to be set before any target that needs it
# We need to use LLVM_INCLUDE_DIRS here, because if we are linking to an
# llvm build directory, this includes $src/llvm/include which is where all the
# headers are not $build/include/ which is what LLVM_INCLUDE_DIR is set to.
include_directories( ${LLVM_INCLUDE_DIRS} )

# Setup prepare_builtins tools
set(LLVM_LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  IRReader
  Support
)
if( LIBCLC_STANDALONE_BUILD )
  add_llvm_executable( prepare_builtins utils/prepare-builtins.cpp )
  set( prepare_builtins_exe prepare_builtins )
  set( prepare_builtins_target prepare_builtins )
else()
  add_llvm_utility( prepare_builtins utils/prepare-builtins.cpp )
  setup_host_tool( prepare_builtins PREPARE_BUILTINS prepare_builtins_exe prepare_builtins_target )
endif()
target_compile_definitions( prepare_builtins PRIVATE ${LLVM_VERSION_DEFINE} )
# These were not properly reported in early LLVM and we don't need them
target_compile_options( prepare_builtins PRIVATE -fno-rtti -fno-exceptions )

# Setup arch devices
set( r600--_devices cedar cypress barts cayman )
set( amdgcn--_devices tahiti )
set( amdgcn-mesa-mesa3d_devices ${amdgcn--_devices} )
set( amdgcn--amdhsa_devices none )
set( clspv--_devices none )
set( clspv64--_devices none )
set( nvptx--_devices none )
set( nvptx64--_devices none )
set( nvptx--nvidiacl_devices none )
set( nvptx64--nvidiacl_devices none )
set( spirv-mesa3d-_devices none )
set( spirv64-mesa3d-_devices none )

# Setup aliases
set( cedar_aliases palm sumo sumo2 redwood juniper )
set( cypress_aliases hemlock )
set( barts_aliases turks caicos )
set( cayman_aliases aruba )
set( tahiti_aliases pitcairn verde oland hainan bonaire kabini kaveri hawaii
  mullins tonga tongapro iceland carrizo fiji stoney polaris10 polaris11
  gfx602 gfx705 gfx805
  gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx90a gfx90c gfx940 gfx941 gfx942
  gfx1010 gfx1011 gfx1012 gfx1013
  gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
  gfx1100 gfx1101 gfx1102 gfx1103
  gfx1150 gfx1151 gfx1152
  gfx1200 gfx1201
)

# pkg-config file
configure_file( libclc.pc.in libclc.pc @ONLY )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/libclc.pc DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig" )
install( DIRECTORY generic/include/clc DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )

if( ENABLE_RUNTIME_SUBNORMAL )
  foreach( file IN ITEMS subnormal_use_default subnormal_disable )
    link_bc(
       TARGET ${file}
       INPUTS ${CMAKE_CURRENT_SOURCE_DIR}/generic/lib/${file}.ll
    )
    install(
      FILES $<TARGET_PROPERTY:${file},TARGET_FILE>
      DESTINATION "${CMAKE_INSTALL_DATADIR}/clc"
    )
  endforeach()
endif()

find_package( Python3 REQUIRED COMPONENTS Interpreter )
file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/generic/lib/gen_convert.py script_loc )
add_custom_command(
  OUTPUT convert.cl
  COMMAND ${Python3_EXECUTABLE} ${script_loc} > convert.cl
  DEPENDS ${script_loc} )
add_custom_target( "generate_convert.cl" DEPENDS convert.cl )
set_target_properties( "generate_convert.cl" PROPERTIES FOLDER "libclc/Sourcegenning" )

add_custom_command(
  OUTPUT clspv-convert.cl
  COMMAND ${Python3_EXECUTABLE} ${script_loc} --clspv > clspv-convert.cl
  DEPENDS ${script_loc} )
add_custom_target( "clspv-generate_convert.cl" DEPENDS clspv-convert.cl )
set_target_properties( "clspv-generate_convert.cl" PROPERTIES FOLDER "libclc/Sourcegenning" )

enable_testing()

foreach( t ${LIBCLC_TARGETS_TO_BUILD} )
  message( STATUS "libclc target '${t}' is enabled" )
  string( REPLACE "-" ";" TRIPLE  ${t} )
  list( GET TRIPLE 0 ARCH )
  list( GET TRIPLE 1 VENDOR )
  list( GET TRIPLE 2 OS )

  set( dirs )

  if ( NOT ${ARCH} STREQUAL spirv AND NOT ${ARCH} STREQUAL spirv64 AND
       NOT ${ARCH} STREQUAL clspv AND NOT ${ARCH} STREQUAL clspv64)
    LIST( APPEND dirs generic )
  endif()

  if( ${ARCH} STREQUAL r600 OR ${ARCH} STREQUAL amdgcn )
    list( APPEND dirs amdgpu )
  endif()

  # nvptx is special
  if( ${ARCH} STREQUAL nvptx OR ${ARCH} STREQUAL nvptx64 )
    set( DARCH ptx )
  else()
    set( DARCH ${ARCH} )
  endif()

  # Enumerate SOURCES* files
  set( source_list )
  foreach( l ${dirs} ${DARCH} ${DARCH}-${OS} ${DARCH}-${VENDOR}-${OS} )
    foreach( s "SOURCES" "SOURCES_${LLVM_MAJOR}.${LLVM_MINOR}" )
      file( TO_CMAKE_PATH ${l}/lib/${s} file_loc )
      file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${file_loc} loc )
      # Prepend the location to give higher priority to
      # specialized implementation
      if( EXISTS ${loc} )
        set( source_list ${file_loc} ${source_list} )
      endif()
    endforeach()
  endforeach()

  # Add the generated convert.cl here to prevent adding the one listed in
  # SOURCES
  set( objects )   # A "set" of already-added input files
  set( rel_files ) # Source directory input files, relative to the root dir
  set( gen_files ) # Generated binary input files, relative to the binary dir
  if( NOT ${ARCH} STREQUAL "spirv" AND NOT ${ARCH} STREQUAL "spirv64" )
    if( NOT ENABLE_RUNTIME_SUBNORMAL AND NOT ${ARCH} STREQUAL "clspv" AND
        NOT ${ARCH} STREQUAL "clspv64" )
      list( APPEND gen_files convert.cl )
      list( APPEND objects convert.cl )
      list( APPEND rel_files generic/lib/subnormal_use_default.ll )
    elseif(${ARCH} STREQUAL "clspv" OR ${ARCH} STREQUAL "clspv64")
      list( APPEND gen_files clspv-convert.cl )
      list( APPEND objects clspv-convert.cl )
    endif()
  endif()

  foreach( l ${source_list} )
    file( READ ${l} file_list )
    string( REPLACE "\n" ";" file_list ${file_list} )
    get_filename_component( dir ${l} DIRECTORY )
    foreach( f ${file_list} )
      # Only add each file once, so that targets can 'specialize' builtins
      if( NOT ${f} IN_LIST objects )
        list( APPEND objects ${f} )
        list( APPEND rel_files ${dir}/${f} )
      endif()
    endforeach()
  endforeach()

  foreach( d ${${t}_devices} )
    get_libclc_device_info(
      TRIPLE ${t}
      DEVICE ${d}
      CPU cpu
      ARCH_SUFFIX arch_suffix
      CLANG_TRIPLE clang_triple
    )

    set( mcpu )
    if( NOT "${cpu}" STREQUAL "" )
      set( mcpu "-mcpu=${cpu}" )
    endif()

    message( STATUS "  device: ${d} ( ${${d}_aliases} )" )

    if ( ARCH STREQUAL spirv OR ARCH STREQUAL spirv64 )
      set( build_flags -O0 -finline-hint-functions )
      set( opt_flags )
      set( spvflags --spirv-max-version=1.1 )
    elseif( ARCH STREQUAL clspv OR ARCH STREQUAL clspv64 )
      set( build_flags "-Wno-unknown-assumption")
      set( opt_flags -O3 )
    else()
      set( build_flags )
      set( opt_flags -O3 )
    endif()

    set( LIBCLC_ARCH_OBJFILE_DIR "${LIBCLC_OBJFILE_DIR}/${arch_suffix}" )
    file( MAKE_DIRECTORY ${LIBCLC_ARCH_OBJFILE_DIR} )

    string( TOUPPER "CLC_${ARCH}" CLC_TARGET_DEFINE )

    list( APPEND build_flags
      -D__CLC_INTERNAL
      -D${CLC_TARGET_DEFINE}
      -I${CMAKE_CURRENT_SOURCE_DIR}/generic/include
      # FIXME: Fix libclc to not require disabling this noisy warning
      -Wno-bitwise-conditional-parentheses
    )

    set( bytecode_files "" )
    foreach( file IN LISTS gen_files rel_files )
      # We need to take each file and produce an absolute input file, as well
      # as a unique architecture-specific output file. We deal with a mix of
      # different input files, which makes this trickier.
      if( ${file} IN_LIST gen_files )
        # Generated files are given just as file names, which we must make
        # absolute to the binary directory.
        set( input_file ${CMAKE_CURRENT_BINARY_DIR}/${file} )
        set( output_file "${LIBCLC_ARCH_OBJFILE_DIR}/${file}.bc" )
      else()
        # Other files are originally relative to each SOURCE file, which are
        # then make relative to the libclc root directory. We must normalize
        # the path (e.g., ironing out any ".."), then make it relative to the
        # root directory again, and use that relative path component for the
        # binary path.
        get_filename_component( abs_path ${file} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
        file( RELATIVE_PATH root_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${abs_path} )
        set( input_file ${CMAKE_CURRENT_SOURCE_DIR}/${file} )
        set( output_file "${LIBCLC_ARCH_OBJFILE_DIR}/${root_rel_path}.bc" )
      endif()

      get_filename_component( file_dir ${file} DIRECTORY )

      compile_to_bc(
        TRIPLE ${clang_triple}
        INPUT ${input_file}
        OUTPUT ${output_file}
        EXTRA_OPTS "${mcpu}" -fno-builtin -nostdlib
                   "${build_flags}" -I${CMAKE_CURRENT_SOURCE_DIR}/${file_dir}
        DEPENDENCIES generate_convert.cl clspv-generate_convert.cl
      )
      list( APPEND bytecode_files ${output_file} )
    endforeach()

    set( builtins_comp_lib_tgt builtins.comp.${arch_suffix} )
    add_custom_target( ${builtins_comp_lib_tgt}
      DEPENDS ${bytecode_files}
    )
    set_target_properties( ${builtins_comp_lib_tgt} PROPERTIES FOLDER "libclc/Device IR/Comp" )

    set( builtins_link_lib_tgt builtins.link.${arch_suffix} )
    link_bc(
      TARGET ${builtins_link_lib_tgt}
      INPUTS ${bytecode_files}
      DEPENDENCIES ${builtins_comp_lib_tgt}
    )

    set( builtins_link_lib $<TARGET_PROPERTY:${builtins_link_lib_tgt},TARGET_FILE> )

    if( ARCH STREQUAL spirv OR ARCH STREQUAL spirv64 )
      set( spv_suffix ${arch_suffix}.spv )
      add_custom_command( OUTPUT ${spv_suffix}
        COMMAND ${llvm-spirv_exe} ${spvflags} -o ${spv_suffix} ${builtins_link_lib}
        DEPENDS ${llvm-spirv_target} ${builtins_link_lib} ${builtins_link_lib_tgt}
      )
      add_custom_target( "prepare-${spv_suffix}" ALL DEPENDS "${spv_suffix}" )
      set_target_properties( "prepare-${spv_suffix}" PROPERTIES FOLDER "libclc/Device IR/Prepare" )
      install( FILES ${CMAKE_CURRENT_BINARY_DIR}/${spv_suffix}
         DESTINATION "${CMAKE_INSTALL_DATADIR}/clc" )
    else()
      set( builtins_opt_lib_tgt builtins.opt.${arch_suffix} )

      # Add opt target
      add_custom_command( OUTPUT ${builtins_opt_lib_tgt}.bc
        COMMAND ${opt_exe} ${opt_flags} -o ${builtins_opt_lib_tgt}.bc
          ${builtins_link_lib}
        DEPENDS ${opt_target} ${builtins_link_lib} ${builtins_link_lib_tgt}
      )
      add_custom_target( ${builtins_opt_lib_tgt}
        ALL DEPENDS ${builtins_opt_lib_tgt}.bc
      )
      set_target_properties( ${builtins_opt_lib_tgt} PROPERTIES
        TARGET_FILE ${CMAKE_CURRENT_BINARY_DIR}/${builtins_opt_lib_tgt}.bc
        FOLDER "libclc/Device IR/Opt"
      )

      set( builtins_opt_lib $<TARGET_PROPERTY:${builtins_opt_lib_tgt},TARGET_FILE> )

      # Add prepare target
      set( obj_suffix ${arch_suffix}.bc )
      add_custom_command( OUTPUT ${obj_suffix}
        COMMAND ${prepare_builtins_exe} -o ${obj_suffix} ${builtins_opt_lib}
        DEPENDS ${builtins_opt_lib} ${builtins_opt_lib_tgt} ${prepare_builtins_target} )
      add_custom_target( prepare-${obj_suffix} ALL DEPENDS ${obj_suffix} )
      set_target_properties( "prepare-${obj_suffix}" PROPERTIES FOLDER "libclc/Device IR/Prepare" )

      # nvptx-- targets don't include workitem builtins
      if( NOT clang_triple MATCHES ".*ptx.*--$" )
        add_test( NAME external-calls-${obj_suffix}
          COMMAND ./check_external_calls.sh ${CMAKE_CURRENT_BINARY_DIR}/${obj_suffix} ${LLVM_TOOLS_BINARY_DIR}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
      endif()

      install( FILES ${CMAKE_CURRENT_BINARY_DIR}/${obj_suffix} DESTINATION "${CMAKE_INSTALL_DATADIR}/clc" )
      foreach( a ${${d}_aliases} )
        set( alias_suffix "${a}-${clang_triple}.bc" )
        add_custom_target( ${alias_suffix} ALL
          COMMAND ${CMAKE_COMMAND} -E create_symlink ${obj_suffix} ${alias_suffix}
          DEPENDS prepare-${obj_suffix} )
        set_target_properties( "${alias_suffix}" PROPERTIES FOLDER "libclc/Device IR/Aliases" )
        install( FILES ${CMAKE_CURRENT_BINARY_DIR}/${alias_suffix} DESTINATION "${CMAKE_INSTALL_DATADIR}/clc" )
      endforeach( a )
    endif()
  endforeach( d )
endforeach( t )
