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

  # If this is an LLVM IR file (identified soley by its file suffix),
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
# * TARGET <string>
#     Custom target to create
# * INPUT <string> ...
#     List of bytecode files to link together
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
function(link_bc)
  cmake_parse_arguments(ARG
    ""
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
    file( WRITE ${RSP_FILE} ${RSP_INPUT} )
    # Ensure that if this file is removed, we re-run CMake
    set_property( DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${RSP_FILE}
    )
    set( LINK_INPUT_ARG "@${RSP_FILE}" )
  endif()

  add_custom_command(
    OUTPUT ${ARG_TARGET}.bc
    COMMAND ${llvm-link_exe} -o ${ARG_TARGET}.bc ${LINK_INPUT_ARG}
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
