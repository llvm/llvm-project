cmake_minimum_required( VERSION 3.9.2 )

project( libclc VERSION 0.2.0 LANGUAGES CXX )
include( GNUInstallDirs )

# List of all targets
set( LIBCLC_TARGETS_ALL
  amdgcn--
  amdgcn--amdhsa
  r600--
  nvptx--
  nvptx64--
  nvptx--nvidiacl
  nvptx64--nvidiacl
)

set( LIBCLC_MIN_LLVM "3.9.0" )

set( LIBCLC_TARGETS_TO_BUILD "all"
    CACHE STRING "Semicolon-separated list of targets to build, or 'all'." )

option( ENABLE_RUNTIME_SUBNORMAL "Enable runtime linking of subnormal support."
OFF )

if( NOT LLVM_CONFIG )
	find_program( LLVM_CONFIG llvm-config )
endif()
execute_process( COMMAND ${LLVM_CONFIG} "--version"
	         OUTPUT_VARIABLE LLVM_VERSION
		 OUTPUT_STRIP_TRAILING_WHITESPACE )
message( "LLVM version: ${LLVM_VERSION}" )

if( ${LLVM_VERSION} VERSION_LESS ${LIBCLC_MIN_LLVM} )
	message( FATAL_ERROR "libclc needs at least LLVM ${LIBCLC_MIN_LLVM}" )
endif()

# mesa3d environment is only available since LLVM 4.0
if( ${LLVM_VERSION} VERSION_GREATER "3.9.0" )
	set( LIBCLC_TARGETS_ALL ${LIBCLC_TARGETS_ALL} amdgcn-mesa-mesa3d )
endif()

if( LIBCLC_TARGETS_TO_BUILD STREQUAL "all" )
	set( LIBCLC_TARGETS_TO_BUILD ${LIBCLC_TARGETS_ALL} )
endif()

list( SORT LIBCLC_TARGETS_TO_BUILD )

execute_process( COMMAND ${LLVM_CONFIG} "--system-libs"
	OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
	OUTPUT_STRIP_TRAILING_WHITESPACE )
execute_process( COMMAND ${LLVM_CONFIG} "--libs" "core" "bitreader" "bitwriter"
	OUTPUT_VARIABLE LLVM_LIBS
	OUTPUT_STRIP_TRAILING_WHITESPACE )
execute_process( COMMAND ${LLVM_CONFIG} "--libdir"
	OUTPUT_VARIABLE LLVM_LIBDIR
	OUTPUT_STRIP_TRAILING_WHITESPACE )
execute_process( COMMAND ${LLVM_CONFIG} "--ldflags"
	OUTPUT_VARIABLE LLVM_LD_FLAGS
	OUTPUT_STRIP_TRAILING_WHITESPACE )
execute_process( COMMAND ${LLVM_CONFIG} "--cxxflags"
	OUTPUT_VARIABLE LLVM_CXX_FLAGS
	OUTPUT_STRIP_TRAILING_WHITESPACE )
separate_arguments( LLVM_CXX_FLAGS )
execute_process( COMMAND ${LLVM_CONFIG} "--bindir"
	OUTPUT_VARIABLE LLVM_BINDIR
	OUTPUT_STRIP_TRAILING_WHITESPACE )

# These were not properly reported in early LLVM and we don't need them
set( LLVM_CXX_FLAGS ${LLVM_CXX_FLAGS} -fno-rtti -fno-exceptions )

# Print LLVM variables
message( "LLVM system libs: ${LLVM_SYSTEM_LIBS}" )
message( "LLVM libs: ${LLVM_LIBS}" )
message( "LLVM libdir: ${LLVM_LIBDIR}" )
message( "LLVM bindir: ${LLVM_BINDIR}" )
message( "LLVM ld flags: ${LLVM_LD_FLAGS}" )
message( "LLVM cxx flags: ${LLVM_CXX_FLAGS}" )
message( "" )

find_program( LLVM_CLANG clang PATHS ${LLVM_BINDIR} NO_DEFAULT_PATH )
find_program( LLVM_AS llvm-as PATHS ${LLVM_BINDIR} NO_DEFAULT_PATH )
find_program( LLVM_LINK llvm-link PATHS ${LLVM_BINDIR} NO_DEFAULT_PATH )
find_program( LLVM_OPT opt PATHS ${LLVM_BINDIR} NO_DEFAULT_PATH )

# Print toolchain
message( "clang: ${LLVM_CLANG}" )
message( "llvm-as: ${LLVM_AS}" )
message( "llvm-link: ${LLVM_LINK}" )
message( "opt: ${LLVM_OPT}" )
message( "" )
if( NOT LLVM_CLANG OR NOT LLVM_OPT OR NOT LLVM_AS OR NOT LLVM_LINK )
	message( FATAL_ERROR "toolchain incomplete!" )
endif()

set( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
set( CMAKE_CLC_COMPILER ${LLVM_CLANG} )
set( CMAKE_CLC_ARCHIVE ${LLVM_LINK} )
set( CMAKE_LLAsm_PREPROCESSOR ${LLVM_CLANG} )
set( CMAKE_LLAsm_COMPILER ${LLVM_AS} )
set( CMAKE_LLAsm_ARCHIVE ${LLVM_LINK} )
enable_language( CLC LLAsm )

# Construct LLVM version define
string( REPLACE "." ";" LLVM_VERSION_LIST ${LLVM_VERSION} )
list( GET LLVM_VERSION_LIST 0 LLVM_MAJOR )
list( GET LLVM_VERSION_LIST 1 LLVM_MINOR )
set( LLVM_VERSION_DEFINE "-DHAVE_LLVM=0x${LLVM_MAJOR}0${LLVM_MINOR}" )

# This needs to be set before any target that needs it
link_directories( ${LLVM_LIBDIR} )

# Setup prepare_builtins tools
add_executable( prepare_builtins utils/prepare-builtins.cpp )
target_compile_options( prepare_builtins PRIVATE ${LLVM_CXX_FLAGS} )
target_compile_definitions( prepare_builtins PRIVATE ${LLVM_VERSION_DEFINE} )
target_link_libraries( prepare_builtins PRIVATE ${LLVM_SYSTEM_LIBS} )
target_link_libraries( prepare_builtins PRIVATE ${LLVM_LIBS} )

# Setup arch devices
set( r600--_devices cedar cypress barts cayman )
set( amdgcn--_devices tahiti )
set( amdgcn-mesa-mesa3d_devices ${amdgcn--_devices} )
set( amdgcn--amdhsa_devices none )
set( nvptx--_devices none )
set( nvptx64--_devices none )
set( nvptx--nvidiacl_devices none )
set( nvptx64--nvidiacl_devices none )

# Setup aliases
set( cedar_aliases palm sumo sumo2 redwood juniper )
set( cypress_aliases hemlock )
set( barts_aliases turks caicos )
set( cayman_aliases aruba )
set( tahiti_aliases pitcairn verde oland hainan bonaire kabini kaveri hawaii
	mullins tonga iceland carrizo fiji stoney polaris10 polaris11 )

# Support for gfx9 was added in LLVM 5.0 (r295554)
if( ${LLVM_VERSION} VERSION_GREATER "4.99.99" )
	set( tahiti_aliases ${tahiti_aliases} gfx900 gfx902 )
endif()

# Support for Vega12 and Vega20 was added in LLVM 7 (r331215)
if( ${LLVM_VERSION} VERSION_GREATER "6.99.99" )
	set( tahiti_aliases ${tahiti_aliases} gfx904 gfx906 )
endif()

# pkg-config file
configure_file( libclc.pc.in libclc.pc @ONLY )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/libclc.pc DESTINATION ${CMAKE_INSTALL_DATADIR}/pkgconfig )
install( DIRECTORY generic/include/clc DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )

if( ENABLE_RUNTIME_SUBNORMAL )
	add_library( subnormal_use_default STATIC
		generic/lib/subnormal_use_default.ll )
	add_library( subnormal_disable STATIC
		generic/lib/subnormal_disable.ll )
	install( TARGETS subnormal_use_default subnormal_disable ARCHIVE
		DESTINATION ${CMAKE_INSTALL_DATADIR}/clc )
endif()

find_program( PYTHON python )
file( TO_CMAKE_PATH ${CMAKE_SOURCE_DIR}/generic/lib/gen_convert.py script_loc )
add_custom_command(
	OUTPUT convert.cl
	COMMAND ${PYTHON} ${script_loc} > convert.cl
	DEPENDS ${script_loc} )
add_custom_target( "generate_convert.cl" DEPENDS convert.cl )

enable_testing()

foreach( t ${LIBCLC_TARGETS_TO_BUILD} )
	message( "BUILDING ${t}" )
	string( REPLACE "-" ";" TRIPLE  ${t} )
	list( GET TRIPLE 0 ARCH )
	list( GET TRIPLE 1 VENDOR )
	list( GET TRIPLE 2 OS )

	set( dirs generic )
	if( ${ARCH} STREQUAL r600 OR ${ARCH} STREQUAL amdgcn )
		set( dirs ${dirs} amdgpu )
	endif()

	#nvptx is special
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
			file( TO_CMAKE_PATH ${CMAKE_SOURCE_DIR}/${file_loc} loc )
			# Prepend the location to give higher priority to
			# specialized implementation
			if( EXISTS ${loc} )
				set( source_list ${file_loc} ${source_list} )
			endif()
		endforeach()
	endforeach()

	# Add the generated convert.cl here to prevent adding
	# the one listed in SOURCES
	set( rel_files convert.cl )
	set( objects convert.cl )
	if( NOT ENABLE_RUNTIME_SUBNORMAL )
		list( APPEND rel_files generic/lib/subnormal_use_default.ll )
	endif()

	foreach( l ${source_list} )
		file( READ ${l} file_list )
		string( REPLACE "\n" ";" file_list ${file_list} )
		get_filename_component( dir ${l} DIRECTORY )
		foreach( f ${file_list} )
			list( FIND objects ${f} found )
			if( found EQUAL  -1 )
				list( APPEND objects ${f} )
				list( APPEND rel_files ${dir}/${f} )
				# FIXME: This should really go away
				file( TO_CMAKE_PATH ${CMAKE_SOURCE_DIR}/${dir}/${f} src_loc )
				get_filename_component( fdir ${src_loc} DIRECTORY )

				set_source_files_properties( ${dir}/${f}
					PROPERTIES COMPILE_FLAGS "-I ${fdir}" )
			endif()
		endforeach()
	endforeach()

	foreach( d ${${t}_devices} )
		# Some targets don't have a specific GPU to target
		if( ${d} STREQUAL "none" )
			set( mcpu )
			set( arch_suffix "${t}" )
		else()
			set( mcpu "-mcpu=${d}" )
			set( arch_suffix "${d}-${t}" )
		endif()
		message( "	DEVICE: ${d} ( ${${d}_aliases} )" )

		add_library( builtins.link.${arch_suffix} STATIC ${rel_files} )
		# Make sure we depend on the pseudo target to prevent
		# multiple invocations
		add_dependencies( builtins.link.${arch_suffix}
			generate_convert.cl )
		# CMake will turn this include into absolute path
		target_include_directories( builtins.link.${arch_suffix} PRIVATE
			"generic/include" )
		target_compile_definitions( builtins.link.${arch_suffix} PRIVATE
			"__CLC_INTERNAL" )
		target_compile_options( builtins.link.${arch_suffix} PRIVATE  -target
			${t} ${mcpu} -fno-builtin )
		set_target_properties( builtins.link.${arch_suffix} PROPERTIES
			LINKER_LANGUAGE CLC )

		set( obj_suffix ${arch_suffix}.bc )

		# Add opt target
		add_custom_command( OUTPUT "builtins.opt.${obj_suffix}"
			            COMMAND ${LLVM_OPT} -O3 -o
				    "builtins.opt.${obj_suffix}"
				    "builtins.link.${obj_suffix}"
				    DEPENDS "builtins.link.${arch_suffix}" )
		add_custom_target( "opt.${obj_suffix}" ALL
		                   DEPENDS "builtins.opt.${obj_suffix}" )

		# Add prepare target
		add_custom_command( OUTPUT "${obj_suffix}"
			            COMMAND prepare_builtins -o
				    "${obj_suffix}"
				    "builtins.opt.${obj_suffix}"
				    DEPENDS "opt.${obj_suffix}"
				            "builtins.opt.${obj_suffix}"
				            prepare_builtins )
		add_custom_target( "prepare-${obj_suffix}" ALL
		                   DEPENDS "${obj_suffix}" )
		install( FILES ${CMAKE_CURRENT_BINARY_DIR}/${obj_suffix} DESTINATION ${CMAKE_INSTALL_DATADIR}/clc )
		# nvptx-- targets don't include workitem builtins
		if( NOT ${t} MATCHES ".*ptx.*--$" )
			add_test( NAME external-calls-${obj_suffix}
				  COMMAND ./check_external_calls.sh ${CMAKE_CURRENT_BINARY_DIR}/${obj_suffix}
				  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} )
			set_tests_properties( external-calls-${obj_suffix}
				PROPERTIES ENVIRONMENT "LLVM_CONFIG=${LLVM_CONFIG}" )
		endif()


		foreach( a ${${d}_aliases} )
			set( alias_suffix "${a}-${t}.bc" )
			add_custom_target( ${alias_suffix} ALL
					   COMMAND ${CMAKE_COMMAND} -E
					   create_symlink ${obj_suffix}
					   ${alias_suffix}
			                   DEPENDS "prepare-${obj_suffix}" )
			install( FILES ${CMAKE_CURRENT_BINARY_DIR}/${alias_suffix} DESTINATION ${CMAKE_INSTALL_DATADIR}/clc )
		endforeach( a )
	endforeach( d )
endforeach( t )
