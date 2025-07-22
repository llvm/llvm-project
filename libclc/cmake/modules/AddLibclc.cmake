# Compiles an OpenCL C - or assembles an LL file - to bytecode
#
# Arguments:
# * TARGET <string>
#     Custom target to create
# * TRIPLE <string>
#     Target triple for which to compile the bytecode file.
# * INPUT <string>
#     File to compile/assemble to bytecode
# * OUTPUT <string>
#     Bytecode file to generate
# * EXTRA_OPTS <string> ...
#     List of compiler options to use. Note that some are added by default.
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
#
# Depends on the clang, llvm-as, and llvm-link targets for compiling,
# assembling, and linking, respectively.
function(compile_to_bc)
  cmake_parse_arguments(ARG
    ""
    "TARGET;TRIPLE;INPUT;OUTPUT"
    "EXTRA_OPTS;DEPENDENCIES"
    ${ARGN}
  )

  # If this is an LLVM IR file (identified solely by its file suffix),
  # pre-process it with clang to a temp file, then assemble that to bytecode.
  set( TMP_SUFFIX )
  get_filename_component( FILE_EXT ${ARG_INPUT} EXT )
  if( NOT ${FILE_EXT} STREQUAL ".ll" )
    # Pass '-c' when not running the preprocessor
    set( PP_OPTS -c )
  else()
    set( PP_OPTS -E;-P )
    set( TMP_SUFFIX .tmp )
  endif()

  set( TARGET_ARG )
  if( ARG_TRIPLE )
    set( TARGET_ARG "-target" ${ARG_TRIPLE} )
  endif()

  # Ensure the directory we are told to output to exists
  get_filename_component( ARG_OUTPUT_DIR ${ARG_OUTPUT} DIRECTORY )
  file( MAKE_DIRECTORY ${ARG_OUTPUT_DIR} )

  add_custom_command(
    OUTPUT ${ARG_OUTPUT}${TMP_SUFFIX}
    COMMAND ${clang_exe}
      ${TARGET_ARG}
      ${PP_OPTS}
      ${ARG_EXTRA_OPTS}
      -MD -MF ${ARG_OUTPUT}.d -MT ${ARG_OUTPUT}${TMP_SUFFIX}
      # LLVM 13 enables standard includes by default - we don't want
      # those when pre-processing IR. We disable it unconditionally.
      $<$<VERSION_GREATER_EQUAL:${LLVM_PACKAGE_VERSION},13.0.0>:-cl-no-stdinc>
      -emit-llvm
      -o ${ARG_OUTPUT}${TMP_SUFFIX}
      -x cl
      ${ARG_INPUT}
    DEPENDS
      ${clang_target}
      ${ARG_INPUT}
      ${ARG_DEPENDENCIES}
    DEPFILE ${ARG_OUTPUT}.d
  )
  # FIXME: The target is added to ensure the parallel build of source files.
  # However, this may result in a large number of targets.
  # Starting with CMake 3.27, DEPENDS_EXPLICIT_ONLY can be used with
  # add_custom_command to enable parallel build.
  # Refer to https://gitlab.kitware.com/cmake/cmake/-/issues/17097 for details.
  add_custom_target( ${ARG_TARGET} DEPENDS ${ARG_OUTPUT}${TMP_SUFFIX} )

  if( ${FILE_EXT} STREQUAL ".ll" )
    add_custom_command(
      OUTPUT ${ARG_OUTPUT}
      COMMAND ${llvm-as_exe} -o ${ARG_OUTPUT} ${ARG_OUTPUT}${TMP_SUFFIX}
      DEPENDS ${llvm-as_target} ${ARG_OUTPUT}${TMP_SUFFIX}
    )
  endif()
endfunction()

# Links together one or more bytecode files
#
# Arguments:
# * INTERNALIZE
#     Set if -internalize flag should be passed when linking
# * TARGET <string>
#     Custom target to create
# * INPUT <string> ...
#     List of bytecode files to link together
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
function(link_bc)
  cmake_parse_arguments(ARG
    "INTERNALIZE"
    "TARGET"
    "INPUTS;DEPENDENCIES"
    ${ARGN}
  )

  set( LINK_INPUT_ARG ${ARG_INPUTS} )
  if( WIN32 OR CYGWIN )
    # Create a response file in case the number of inputs exceeds command-line
    # character limits on certain platforms.
    file( TO_CMAKE_PATH ${LIBCLC_ARCH_OBJFILE_DIR}/${ARG_TARGET}.rsp RSP_FILE )
    # Turn it into a space-separate list of input files
    list( JOIN ARG_INPUTS " " RSP_INPUT )
    file( GENERATE OUTPUT ${RSP_FILE} CONTENT ${RSP_INPUT} )
    # Ensure that if this file is removed, we re-run CMake
    set_property( DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${RSP_FILE}
    )
    set( LINK_INPUT_ARG "@${RSP_FILE}" )
  endif()

  if( ARG_INTERNALIZE )
    set( link_flags --internalize --only-needed )
  endif()

  add_custom_command(
    OUTPUT ${ARG_TARGET}.bc
    COMMAND ${llvm-link_exe} ${link_flags} -o ${ARG_TARGET}.bc ${LINK_INPUT_ARG}
    DEPENDS ${llvm-link_target} ${ARG_DEPENDENCIES} ${ARG_INPUTS} ${RSP_FILE}
  )

  add_custom_target( ${ARG_TARGET} ALL DEPENDS ${ARG_TARGET}.bc )
  set_target_properties( ${ARG_TARGET} PROPERTIES
    TARGET_FILE ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}.bc
    FOLDER "libclc/Device IR/Linking"
  )
endfunction()

# Decomposes and returns variables based on a libclc triple and architecture
# combination. Returns data via one or more optional output variables.
#
# Arguments:
# * TRIPLE <string>
#     libclc target triple to query
# * DEVICE <string>
#     libclc device to query
#
# Optional Arguments:
# * CPU <var>
#     Variable name to be set to the target CPU
# * ARCH_SUFFIX <var>
#     Variable name to be set to the triple/architecture suffix
# * CLANG_TRIPLE <var>
#     Variable name to be set to the normalized clang triple
function(get_libclc_device_info)
  cmake_parse_arguments(ARG
    ""
    "TRIPLE;DEVICE;CPU;ARCH_SUFFIX;CLANG_TRIPLE"
    ""
    ${ARGN}
  )

  if( NOT ARG_TRIPLE OR NOT ARG_DEVICE )
    message( FATAL_ERROR "Must provide both TRIPLE and DEVICE" )
  endif()

  string( REPLACE "-" ";" TRIPLE  ${ARG_TRIPLE} )
  list( GET TRIPLE 0 ARCH )

  # Some targets don't have a specific device architecture to target
  if( ARG_DEVICE STREQUAL none OR ARCH STREQUAL spirv OR ARCH STREQUAL spirv64 )
    set( cpu )
    set( arch_suffix "${ARG_TRIPLE}" )
  else()
    set( cpu "${ARG_DEVICE}" )
    set( arch_suffix "${ARG_DEVICE}-${ARG_TRIPLE}" )
  endif()

  if( ARG_CPU )
    set( ${ARG_CPU} ${cpu} PARENT_SCOPE )
  endif()

  if( ARG_ARCH_SUFFIX )
    set( ${ARG_ARCH_SUFFIX} ${arch_suffix} PARENT_SCOPE )
  endif()

  # Some libclc targets are not real clang triples: return their canonical
  # triples.
  if( ARCH STREQUAL spirv OR ARCH STREQUAL clspv )
    set( ARG_TRIPLE "spir--" )
  elseif( ARCH STREQUAL spirv64 OR ARCH STREQUAL clspv64 )
    set( ARG_TRIPLE "spir64--" )
  endif()

  if( ARG_CLANG_TRIPLE )
    set( ${ARG_CLANG_TRIPLE} ${ARG_TRIPLE} PARENT_SCOPE )
  endif()
endfunction()

# Compiles a list of library source files (provided by LIB_FILES/GEN_FILES) and
# compiles them to LLVM bytecode (or SPIR-V), links them together and optimizes
# them.
#
# For bytecode libraries, a list of ALIASES may optionally be provided to
# produce additional symlinks.
#
# Arguments:
#  * ARCH <string>
#      libclc architecture being built
#  * ARCH_SUFFIX <string>
#      libclc architecture/triple suffix
#  * TRIPLE <string>
#      Triple used to compile
#
# Optional Arguments:
# * CLC_INTERNAL
#     Pass if compiling the internal CLC builtin libraries, which are not
#     optimized and do not have aliases created.
#  * LIB_FILES <string> ...
#      List of files that should be built for this library
#  * GEN_FILES <string> ...
#      List of generated files (in build dir) that should be built for this library
#  * COMPILE_FLAGS <string> ...
#      Compilation options (for clang)
#  * OPT_FLAGS <string> ...
#      Optimization options (for opt)
#  * ALIASES <string> ...
#      List of aliases
#  * INTERNAL_LINK_DEPENDENCIES <target> ...
#      A list of extra bytecode file's targets. The bitcode files will be linked
#      into the builtin library. Symbols from these link dependencies will be
#      internalized during linking.
function(add_libclc_builtin_set)
  cmake_parse_arguments(ARG
    "CLC_INTERNAL"
    "ARCH;TRIPLE;ARCH_SUFFIX"
    "LIB_FILES;GEN_FILES;COMPILE_FLAGS;OPT_FLAGS;ALIASES;INTERNAL_LINK_DEPENDENCIES"
    ${ARGN}
  )

  if( NOT ARG_ARCH OR NOT ARG_ARCH_SUFFIX OR NOT ARG_TRIPLE )
    message( FATAL_ERROR "Must provide ARCH, ARCH_SUFFIX, and TRIPLE" )
  endif()

  set( bytecode_files )
  set( bytecode_ir_files )
  set( compile_tgts )
  foreach( file IN LISTS ARG_GEN_FILES ARG_LIB_FILES )
    # We need to take each file and produce an absolute input file, as well
    # as a unique architecture-specific output file. We deal with a mix of
    # different input files, which makes this trickier.
    set( input_file_dep )
    if( ${file} IN_LIST ARG_GEN_FILES )
      # Generated files are given just as file names, which we must make
      # absolute to the binary directory.
      set( input_file ${CMAKE_CURRENT_BINARY_DIR}/${file} )
      set( output_file "${LIBCLC_ARCH_OBJFILE_DIR}/${file}.bc" )
      # If a target exists that generates this file, add that as a dependency
      # of the custom command.
      if( TARGET generate-${file} )
        set( input_file_dep generate-${file} )
      endif()
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

    string( REPLACE "/" "-" replaced ${file} )
    set( tgt compile_tgt-${ARG_ARCH_SUFFIX}${replaced})

    set( file_specific_compile_options )
    get_source_file_property( compile_opts ${file} COMPILE_OPTIONS)
    if( compile_opts )
      set( file_specific_compile_options "${compile_opts}" )
    endif()

    compile_to_bc(
      TARGET ${tgt}
      TRIPLE ${ARG_TRIPLE}
      INPUT ${input_file}
      OUTPUT ${output_file}
      EXTRA_OPTS -fno-builtin -nostdlib "${ARG_COMPILE_FLAGS}"
        "${file_specific_compile_options}"
        -I${CMAKE_CURRENT_SOURCE_DIR}/${file_dir}
      DEPENDENCIES ${input_file_dep}
    )
    list( APPEND compile_tgts ${tgt} )

    # Collect all files originating in LLVM IR separately
    get_filename_component( file_ext ${file} EXT )
    if( ${file_ext} STREQUAL ".ll" )
      list( APPEND bytecode_ir_files ${output_file} )
    else()
      list( APPEND bytecode_files ${output_file} )
    endif()
  endforeach()

  # Prepend all LLVM IR files to the list so they are linked into the final
  # bytecode modules first. This helps to suppress unnecessary warnings
  # regarding different data layouts while linking. Any LLVM IR files without a
  # data layout will (silently) be given the first data layout the linking
  # process comes across.
  list( PREPEND bytecode_files ${bytecode_ir_files} )

  set( builtins_comp_lib_tgt builtins.comp.${ARG_ARCH_SUFFIX} )
  add_custom_target( ${builtins_comp_lib_tgt}
    DEPENDS ${bytecode_files} ${compile_tgts}
  )
  set_target_properties( ${builtins_comp_lib_tgt} PROPERTIES FOLDER "libclc/Device IR/Comp" )

  if( NOT bytecode_files )
    message(FATAL_ERROR "Cannot create an empty builtins library")
  endif()

  set( builtins_link_lib_tgt builtins.link.${ARG_ARCH_SUFFIX} )

  if( NOT ARG_INTERNAL_LINK_DEPENDENCIES )
    link_bc(
      TARGET ${builtins_link_lib_tgt}
      INPUTS ${bytecode_files}
      DEPENDENCIES ${builtins_comp_lib_tgt}
    )
  else()
    # If we have libraries to link while internalizing their symbols, we need
    # two separate link steps; the --internalize flag applies to all link
    # inputs but the first.
    set( builtins_link_lib_tmp_tgt builtins.link.pre-deps.${ARG_ARCH_SUFFIX} )
    link_bc(
      TARGET ${builtins_link_lib_tmp_tgt}
      INPUTS ${bytecode_files}
      DEPENDENCIES ${builtins_comp_lib_tgt}
    )
    set( internal_link_depend_files )
    foreach( tgt ${ARG_INTERNAL_LINK_DEPENDENCIES} )
      list( APPEND internal_link_depend_files $<TARGET_PROPERTY:${tgt},TARGET_FILE> )
    endforeach()
    link_bc(
      INTERNALIZE
      TARGET ${builtins_link_lib_tgt}
      INPUTS $<TARGET_PROPERTY:${builtins_link_lib_tmp_tgt},TARGET_FILE>
        ${internal_link_depend_files}
      DEPENDENCIES ${builtins_link_lib_tmp_tgt} ${ARG_INTERNAL_LINK_DEPENDENCIES}
    )
  endif()

  # For the CLC internal builtins, exit here - we only optimize the targets'
  # entry points once we've linked the CLC buitins into them
  if( ARG_CLC_INTERNAL )
    return()
  endif()

  set( builtins_link_lib $<TARGET_PROPERTY:${builtins_link_lib_tgt},TARGET_FILE> )

  # For SPIR-V targets we diverage at this point and generate SPIR-V using the
  # llvm-spirv tool.
  if( ARG_ARCH STREQUAL spirv OR ARG_ARCH STREQUAL spirv64 )
    set( obj_suffix ${ARG_ARCH_SUFFIX}.spv )
    add_custom_command( OUTPUT ${obj_suffix}
      COMMAND ${llvm-spirv_exe} ${spvflags} -o ${obj_suffix} ${builtins_link_lib}
      DEPENDS ${llvm-spirv_target} ${builtins_link_lib} ${builtins_link_lib_tgt}
    )
  else()
    # Non-SPIR-V targets add an extra step to optimize the bytecode
    set( builtins_opt_lib_tgt builtins.opt.${ARG_ARCH_SUFFIX} )

    add_custom_command( OUTPUT ${builtins_opt_lib_tgt}.bc
      COMMAND ${opt_exe} ${ARG_OPT_FLAGS} -o ${builtins_opt_lib_tgt}.bc
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

    set( obj_suffix ${ARG_ARCH_SUFFIX}.bc )
    add_custom_command( OUTPUT ${obj_suffix}
      COMMAND ${prepare_builtins_exe} -o ${obj_suffix} ${builtins_opt_lib}
      DEPENDS ${builtins_opt_lib} ${builtins_opt_lib_tgt} ${prepare_builtins_target} )
  endif()

  # Add a 'prepare' target
  add_custom_target( prepare-${obj_suffix} ALL DEPENDS ${obj_suffix} )
  set_target_properties( "prepare-${obj_suffix}" PROPERTIES FOLDER "libclc/Device IR/Prepare" )

  # Also add a 'prepare' target for the triple. Since a triple may have
  # multiple devices, ensure we only try to create the triple target once. The
  # triple's target will build all of the bytecode for its constituent devices.
  if( NOT TARGET prepare-${ARG_TRIPLE} )
    add_custom_target( prepare-${ARG_TRIPLE} ALL )
  endif()
  add_dependencies( prepare-${ARG_TRIPLE} prepare-${obj_suffix} )

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${obj_suffix}
    DESTINATION "${CMAKE_INSTALL_DATADIR}/clc"
  )

  # SPIR-V targets can exit early here
  if( ARG_ARCH STREQUAL spirv OR ARG_ARCH STREQUAL spirv64 )
    return()
  endif()

  # Add a test for whether or not the libraries contain unresolved calls which
  # would usually indicate a build problem. Note that we don't perform this
  # test for all libclc targets:
  # * nvptx-- targets don't include workitem builtins
  # * clspv targets don't include all OpenCL builtins
  if( NOT ARG_ARCH MATCHES "^(nvptx|clspv)(64)?$" )
    add_test( NAME external-calls-${obj_suffix}
      COMMAND ./check_external_calls.sh ${CMAKE_CURRENT_BINARY_DIR}/${obj_suffix} ${LLVM_TOOLS_BINARY_DIR}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
  endif()

  foreach( a ${ARG_ALIASES} )
    set( alias_suffix "${a}-${ARG_TRIPLE}.bc" )
    add_custom_command(
      OUTPUT ${alias_suffix}
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${obj_suffix} ${alias_suffix}
      DEPENDS prepare-${obj_suffix} )
    add_custom_target( alias-${alias_suffix} ALL DEPENDS ${alias_suffix} )
    set_target_properties( alias-${alias_suffix} PROPERTIES FOLDER "libclc/Device IR/Aliases" )
    install( FILES ${CMAKE_CURRENT_BINARY_DIR}/${alias_suffix}
             DESTINATION "${CMAKE_INSTALL_DATADIR}/clc" )
  endforeach( a )
endfunction(add_libclc_builtin_set)

# Produces a list of libclc source files by walking over SOURCES files in a
# given directory. Outputs the list of files in LIB_FILE_LIST.
#
# LIB_FILE_LIST may be pre-populated and is appended to.
#
# Arguments:
# * LIB_ROOT_DIR <string>
#     Root directory containing target's lib files, relative to libclc root
#     directory. If not provided, is set to '.'.
# * DIRS <string> ...
#     List of directories under LIB_ROOT_DIR to walk over searching for SOURCES
#     files. Directories earlier in the list have lower priority than
#     subsequent ones.
function(libclc_configure_lib_source LIB_FILE_LIST)
  cmake_parse_arguments(ARG
    ""
    "LIB_ROOT_DIR"
    "DIRS"
    ${ARGN}
  )

  if( NOT ARG_LIB_ROOT_DIR )
    set(ARG_LIB_ROOT_DIR  ".")
  endif()

  # Enumerate SOURCES* files
  set( source_list )
  foreach( l IN LISTS ARG_DIRS )
    foreach( s "SOURCES" "SOURCES_${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}" )
      file( TO_CMAKE_PATH ${ARG_LIB_ROOT_DIR}/lib/${l}/${s} file_loc )
      file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${file_loc} loc )
      # Prepend the location to give higher priority to the specialized
      # implementation
      if( EXISTS ${loc} )
        list( PREPEND source_list ${file_loc} )
      endif()
    endforeach()
  endforeach()

  ## Add the generated convert files here to prevent adding the ones listed in
  ## SOURCES
  set( rel_files ${${LIB_FILE_LIST}} ) # Source directory input files, relative to the root dir
  # A "set" of already-added input files
  set( objects )
  foreach( f ${${LIB_FILE_LIST}} )
    get_filename_component( name ${f} NAME )
    list( APPEND objects ${name} )
  endforeach()

  foreach( l ${source_list} )
    file( READ ${l} file_list )
    string( REPLACE "\n" ";" file_list ${file_list} )
    get_filename_component( dir ${l} DIRECTORY )
    foreach( f ${file_list} )
      get_filename_component( name ${f} NAME )
      # Only add each file once, so that targets can 'specialize' builtins
      if( NOT ${name} IN_LIST objects )
        list( APPEND objects ${name} )
        list( APPEND rel_files ${dir}/${f} )
      endif()
    endforeach()
  endforeach()

  set( ${LIB_FILE_LIST} ${rel_files} PARENT_SCOPE )
endfunction(libclc_configure_lib_source LIB_FILE_LIST)
