# Compiles an OpenCL C - or assembles an LL file - to bytecode
#
# Arguments:
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
    "TRIPLE;INPUT;OUTPUT"
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
    set( EXTRA_OPTS ${ARG_EXTRA_OPTS} )
  else()
    set( PP_OPTS -E;-P )
    set( TMP_SUFFIX .tmp )
    string( REPLACE "-Xclang;-fdeclare-opencl-builtins;-Xclang;-finclude-default-header"
      "" EXTRA_OPTS "${ARG_EXTRA_OPTS}"
    )
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
      ${EXTRA_OPTS}
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
    OUTPUT ${LIBCLC_ARCH_OBJFILE_DIR}/${ARG_TARGET}.bc
    COMMAND ${llvm-link_exe} ${link_flags} -o ${LIBCLC_ARCH_OBJFILE_DIR}/${ARG_TARGET}.bc ${LINK_INPUT_ARG}
    DEPENDS ${llvm-link_target} ${ARG_DEPENDENCIES} ${ARG_INPUTS} ${RSP_FILE}
  )

  add_custom_target( ${ARG_TARGET} ALL DEPENDS ${LIBCLC_ARCH_OBJFILE_DIR}/${ARG_TARGET}.bc )
  set_target_properties( ${ARG_TARGET} PROPERTIES
    TARGET_FILE ${LIBCLC_ARCH_OBJFILE_DIR}/${ARG_TARGET}.bc
    FOLDER "libclc/Device IR/Linking"
  )
endfunction()

# Create a custom target for each bitcode file, which is the output of a custom
# command. This is required for parallel compilation of the custom commands that
# generate the bitcode files when using the CMake MSVC generator on Windows.
#
# Arguments:
#  * compile_tgts
#      Output list of compile targets
#  * ARCH_SUFFIX <string>
#      libclc architecture/triple suffix
#  * FILES <string> ...
#     List of bitcode files
function(create_compile_targets compile_tgts)
  cmake_parse_arguments( ARG "" "ARCH_SUFFIX" "FILES" ${ARGN} )

  if( NOT ARG_ARCH_SUFFIX OR NOT ARG_FILES )
    message( FATAL_ERROR "Must provide ARCH_SUFFIX, and FILES" )
  endif()

  set( tgts )
  foreach( file IN LISTS ARG_FILES )
    cmake_path( GET file STEM stem )
    cmake_path( GET file PARENT_PATH parent_path )
    cmake_path( GET parent_path STEM parent_path_stem )
    set( tgt compile-${ARG_ARCH_SUFFIX}-${parent_path_stem}-${stem} )
    add_custom_target( ${tgt} DEPENDS ${file} )
    list( APPEND tgts ${tgt} )
  endforeach()

  set( compile_tgts ${tgts} PARENT_SCOPE )
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
  if( ARG_DEVICE STREQUAL none
      OR ((ARCH STREQUAL spirv OR ARCH STREQUAL spirv64)
          AND NOT LIBCLC_USE_SPIRV_BACKEND) )
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
  if( ARCH STREQUAL spirv AND LIBCLC_USE_SPIRV_BACKEND )
    set( ARG_TRIPLE "spirv32--" )
  elseif( ARCH STREQUAL spirv64 AND LIBCLC_USE_SPIRV_BACKEND )
    set( ARG_TRIPLE "spirv64--" )
  elseif( ARCH STREQUAL spirv OR ARCH STREQUAL clspv )
    set( ARG_TRIPLE "spir--" )
  elseif( ARCH STREQUAL spirv64 OR ARCH STREQUAL clspv64 )
    set( ARG_TRIPLE "spir64--" )
  endif()

  if( ARG_CLANG_TRIPLE )
    set( ${ARG_CLANG_TRIPLE} ${ARG_TRIPLE} PARENT_SCOPE )
  endif()
endfunction()

# Compiles a list of library source files (provided by LIB_FILES) and compiles
# them to LLVM bytecode (or SPIR-V), links them together and optimizes them.
#
# For bytecode libraries, a list of ALIASES may optionally be provided to
# produce additional symlinks.
#
# Arguments:
#  * ARCH <string>
#      libclc architecture being built
#  * DEVICE <string>
#      libclc microarchitecture being built
#  * ARCH_SUFFIX <string>
#      libclc architecture/triple suffix
#  * TRIPLE <string>
#      Triple used to compile
#  * PARENT_TARGET <string>
#      Target into which to group the target builtins
#
# Optional Arguments:
# * CLC_INTERNAL
#     Pass if compiling the internal CLC builtin libraries, which are not
#     optimized and do not have aliases created.
#  * LIB_FILES <string> ...
#      List of files that should be built for this library
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
    "ARCH;DEVICE;TRIPLE;ARCH_SUFFIX;PARENT_TARGET"
    "LIB_FILES;COMPILE_FLAGS;OPT_FLAGS;ALIASES;INTERNAL_LINK_DEPENDENCIES"
    ${ARGN}
  )

  if( NOT ARG_ARCH OR NOT ARG_ARCH_SUFFIX OR NOT ARG_TRIPLE )
    message( FATAL_ERROR "Must provide ARCH, ARCH_SUFFIX, and TRIPLE" )
  endif()

  set( bytecode_files )
  set( bytecode_ir_files )
  foreach( file IN LISTS ARG_LIB_FILES )
    # Files are originally relative to each SOURCE file, which are then make
    # relative to the libclc root directory. We must normalize the path
    # (e.g., ironing out any ".."), then make it relative to the root directory
    # again, and use that relative path component for the binary path.
    get_filename_component( abs_path ${file} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
    file( RELATIVE_PATH root_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${abs_path} )
    set( input_file ${CMAKE_CURRENT_SOURCE_DIR}/${file} )
    set( output_file "${LIBCLC_ARCH_OBJFILE_DIR}/${root_rel_path}.bc" )

    get_filename_component( file_dir ${file} DIRECTORY )

    set( file_specific_compile_options )
    get_source_file_property( compile_opts ${file} COMPILE_OPTIONS)
    if( compile_opts )
      set( file_specific_compile_options "${compile_opts}" )
    endif()

    compile_to_bc(
      TRIPLE ${ARG_TRIPLE}
      INPUT ${input_file}
      OUTPUT ${output_file}
      EXTRA_OPTS -nostdlib "${ARG_COMPILE_FLAGS}"
        "${file_specific_compile_options}"
        -I${CMAKE_CURRENT_SOURCE_DIR}/${file_dir}
    )

    # Collect all files originating in LLVM IR separately
    get_filename_component( file_ext ${file} EXT )
    if( ${file_ext} STREQUAL ".ll" )
      list( APPEND bytecode_ir_files ${output_file} )
    else()
      list( APPEND bytecode_files ${output_file} )
    endif()
  endforeach()

  set( builtins_comp_lib_tgt builtins.comp.${ARG_ARCH_SUFFIX} )
  if ( CMAKE_GENERATOR MATCHES "Visual Studio" )
    # Don't put commands in one custom target to avoid serialized compilation.
    create_compile_targets( compile_tgts
      ARCH_SUFFIX ${ARG_ARCH_SUFFIX}
      FILES ${bytecode_files}
    )
    add_custom_target( ${builtins_comp_lib_tgt} DEPENDS ${bytecode_ir_files} )
    add_dependencies( ${builtins_comp_lib_tgt} ${compile_tgts} )
  else()
    add_custom_target( ${builtins_comp_lib_tgt}
      DEPENDS ${bytecode_files} ${bytecode_ir_files}
    )
  endif()
  set_target_properties( ${builtins_comp_lib_tgt} PROPERTIES FOLDER "libclc/Device IR/Comp" )

  # Prepend all LLVM IR files to the list so they are linked into the final
  # bytecode modules first. This helps to suppress unnecessary warnings
  # regarding different data layouts while linking. Any LLVM IR files without a
  # data layout will (silently) be given the first data layout the linking
  # process comes across.
  list( PREPEND bytecode_files ${bytecode_ir_files} )

  if( NOT bytecode_files )
    message(FATAL_ERROR "Cannot create an empty builtins library for ${ARG_ARCH_SUFFIX}")
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

  set( LIBCLC_OUTPUT_FILENAME libclc )
  set( builtins_link_lib $<TARGET_PROPERTY:${builtins_link_lib_tgt},TARGET_FILE> )

  # We store the library according to its triple and cpu if present.
  if (NOT "${ARG_DEVICE}" STREQUAL "none")
    set (library_dir ${LIBCLC_OUTPUT_LIBRARY_DIR}/${ARG_TRIPLE}/${ARG_DEVICE})
  else()
    set (library_dir ${LIBCLC_OUTPUT_LIBRARY_DIR}/${ARG_TRIPLE})
  endif()
  file( MAKE_DIRECTORY ${library_dir} )

  # For SPIR-V targets we diverage at this point and generate SPIR-V using the
  # llvm-spirv tool.
  if( ARG_ARCH STREQUAL spirv OR ARG_ARCH STREQUAL spirv64 )
    set( libclc_builtins_lib ${library_dir}/${LIBCLC_OUTPUT_FILENAME}.spv )
    if ( LIBCLC_USE_SPIRV_BACKEND )
      add_custom_command( OUTPUT ${libclc_builtins_lib}
        COMMAND ${clang_exe} -c --target=${ARG_TRIPLE} -x ir -o ${libclc_builtins_lib} ${builtins_link_lib}
        DEPENDS ${clang_target} ${builtins_link_lib} ${builtins_link_lib_tgt}
      )
    else()
      add_custom_command( OUTPUT ${libclc_builtins_lib}
        COMMAND ${llvm-spirv_exe} ${spvflags} -o ${libclc_builtins_lib} ${builtins_link_lib}
        DEPENDS ${llvm-spirv_target} ${builtins_link_lib} ${builtins_link_lib_tgt}
      )
    endif()
  else()
    # Non-SPIR-V targets add an extra step to optimize the bytecode
    set( libclc_builtins_lib ${library_dir}/${LIBCLC_OUTPUT_FILENAME}.bc )

    add_custom_command( OUTPUT ${libclc_builtins_lib}
      COMMAND ${opt_exe} ${ARG_OPT_FLAGS} -o ${libclc_builtins_lib}
        ${builtins_link_lib}
      DEPENDS ${opt_target} ${builtins_link_lib} ${builtins_link_lib_tgt}
    )
  endif()

  # Add a 'library' target
  add_custom_target( library-${ARG_ARCH_SUFFIX} ALL DEPENDS ${libclc_builtins_lib} )
  set_target_properties( "library-${ARG_ARCH_SUFFIX}" PROPERTIES
    TARGET_FILE ${libclc_builtins_lib}
    FOLDER "libclc/Device IR/Library"
  )

  # Also add a 'library' target for the triple. Since a triple may have
  # multiple devices, ensure we only try to create the triple target once. The
  # triple's target will build all of the bytecode for its constituent devices.
  if( NOT TARGET library-${ARG_TRIPLE} )
    add_custom_target( library-${ARG_TRIPLE} ALL )
  endif()
  add_dependencies( library-${ARG_TRIPLE} library-${ARG_ARCH_SUFFIX} )
  # Add dependency to top-level pseudo target to ease making other
  # targets dependent on libclc.
  add_dependencies( ${ARG_PARENT_TARGET} library-${ARG_TRIPLE} )

  # Install the created library.
  install(
    FILES ${libclc_builtins_lib}
    DESTINATION ${LIBCLC_INSTALL_DIR}/${ARG_TRIPLE}
  )

  # SPIR-V targets can exit early here
  if( ARG_ARCH STREQUAL spirv OR ARG_ARCH STREQUAL spirv64 )
    return()
  endif()

  # Add a test for whether or not the libraries contain unresolved functions
  # which would usually indicate a build problem. Note that we don't perform
  # this test for all libclc targets:
  # * nvptx64-- targets don't include workitem builtins
  # * clspv targets don't include all OpenCL builtins
  if( NOT ARG_ARCH MATCHES "^(nvptx|clspv)(64)?$" )
    add_test( NAME external-funcs-${ARG_ARCH_SUFFIX}
      COMMAND ./check_external_funcs.sh ${libclc_builtins_lib} ${LLVM_TOOLS_BINARY_DIR}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
  endif()

  foreach( a IN LISTS ARG_ALIASES )
    if(CMAKE_HOST_UNIX OR LLVM_USE_SYMLINKS)
      cmake_path(RELATIVE_PATH libclc_builtins_lib
        BASE_DIRECTORY ${LIBCLC_OUTPUT_LIBRARY_DIR}
        OUTPUT_VARIABLE LIBCLC_LINK_OR_COPY_SOURCE)
      set(LIBCLC_LINK_OR_COPY create_symlink)
    else()
      set(LIBCLC_LINK_OR_COPY_SOURCE ${libclc_builtins_lib})
      set(LIBCLC_LINK_OR_COPY copy)
    endif()

    file( MAKE_DIRECTORY ${LIBCLC_OUTPUT_LIBRARY_DIR}/${ARG_TRIPLE}/${a} )
    set( libclc_alias_lib ${LIBCLC_OUTPUT_LIBRARY_DIR}/${ARG_TRIPLE}/${a}/${LIBCLC_OUTPUT_FILENAME}.bc )
    add_custom_command(
      OUTPUT ${libclc_alias_lib}
      COMMAND ${CMAKE_COMMAND} -E ${LIBCLC_LINK_OR_COPY} ${LIBCLC_LINK_OR_COPY_SOURCE} ${libclc_alias_lib}
      DEPENDS library-${ARG_ARCH_SUFFIX}
    )
    add_custom_target( alias-${a}-${ARG_TRIPLE} ALL
      DEPENDS ${libclc_alias_lib}
    )
    add_dependencies( ${ARG_PARENT_TARGET} alias-${a}-${ARG_TRIPLE} )
    set_target_properties( alias-${a}-${ARG_TRIPLE}
      PROPERTIES FOLDER "libclc/Device IR/Aliases"
    )

    # Install the library
    install(
      FILES ${libclc_alias_lib}
      DESTINATION ${LIBCLC_INSTALL_DIR}/${ARG_TRIPLE}/${a}
    )
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
