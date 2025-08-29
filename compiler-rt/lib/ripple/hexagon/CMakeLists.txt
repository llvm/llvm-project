
if(NOT (${HEXAGON} IN_LIST RIPPLE_SUPPORTED_ARCH))
  message(FATAL_ERROR "Not compiling. ${RIPPLE_SUPPORTED_ARCH}")
  return()
endif()

# ------------------------------------------------------------------------------
# Function: ripple_add_bc_rules
# Creates one .bc output per source base name by copying the corresponding object
# from the single per-arch object library target.
#
# Usage:
#   ripple_add_bc_rules(
#     <objlib_base>      # e.g. RippleLib.79   (without .${HEXAGON})
#     <hexagon_triple>   # ${HEXAGON}
#     <src_base_names>   # ; separated list of names like: HVX_Scatter_Gather;HVX_Rotate
#     <bc_out_dir>       # directory for .bc files
#     <out_var>          # variable name to receive the list of bc files
#   )
# ------------------------------------------------------------------------------
function(ripple_add_bc_rules OBJLIB_BASE HEXAGON SRC_BASE_NAMES BC_DIR OUT_VAR)
  set(full_tgt "${OBJLIB_BASE}.${HEXAGON}")
  set(all_bcs)

  foreach(_name IN LISTS SRC_BASE_NAMES)
    # IMPORTANT:
    #  - No whitespace inside the $<FILTER:...> argument list.
    #  - Quote the whole genex when passing to COMMAND to keep it as one token.
    set(_obj "$<FILTER:$<TARGET_OBJECTS:${full_tgt}>,INCLUDE,.*/${_name}(\\.(c|cc|cpp|cxx))?\\.(o|obj)$>")
    set(_bc  "${BC_DIR}/${_name}.bc")

    add_custom_command(
      OUTPUT "${_bc}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${BC_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_obj}" "${_bc}"
      DEPENDS $<TARGET_OBJECTS:${full_tgt}>
      COMMENT "Generating ${_name}.bc from ${full_tgt}"
      VERBATIM
      COMMAND_EXPAND_LISTS  # expand any list produced by the genex into args
    )
    list(APPEND all_bcs "${_bc}")
  endforeach()

  set(${OUT_VAR} "${all_bcs}" PARENT_SCOPE)
endfunction()

set(RT_INCLUDES
  -I${CMAKE_CURRENT_SOURCE_DIR}/include
  -I${CMAKE_CURRENT_SOURCE_DIR}/header
)

add_custom_target(RippleLib ALL)

# Keep in sync w/ llvm/lib/Target/Hexagon/Hexagon.td
set(HEXAGON_ARCH_VERSIONS 68 69 71 73 75 79)

set(RIPPLE_RT_SRC HVX_Scatter_Gather HVX_Rotate HVX_Bfloat)
set(RIPPLE_RT_SRC_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(RIPPLE_RT_HEADER_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/header)

foreach(HEXAGON_ARCH_VERSION IN LISTS HEXAGON_ARCH_VERSIONS)
  set(HEXAGON_FLAGS
      -mv${HEXAGON_ARCH_VERSION}
      -mhvx=v${HEXAGON_ARCH_VERSION}
      -mhvx-qfloat)
  set(RIPPLE_FLAG "-fenable-ripple") # only for HVX_Scatter_Gather

  # Resolve actual source files and keep the base names that truly exist
  set(RIPPLE_RT_SOURCES)
  set(RIPPLE_RT_NAMES)  # base names that were found
  foreach(FILE_NAME IN LISTS RIPPLE_RT_SRC)
    set(SRC_FILE "")
    if(EXISTS ${RIPPLE_RT_SRC_DIR}/${FILE_NAME}.cpp)
      set(SRC_FILE ${RIPPLE_RT_SRC_DIR}/${FILE_NAME}.cpp)
    elseif(EXISTS ${RIPPLE_RT_SRC_DIR}/${FILE_NAME}.c)
      set(SRC_FILE ${RIPPLE_RT_SRC_DIR}/${FILE_NAME}.c)
    elseif(EXISTS ${RIPPLE_RT_SRC_DIR}/${FILE_NAME}.cc)
      set(SRC_FILE ${RIPPLE_RT_SRC_DIR}/${FILE_NAME}.cc)
    else()
      message(WARNING "Cannot find source file for ${FILE_NAME}.")
      continue()
    endif()
    list(APPEND RIPPLE_RT_SOURCES "${SRC_FILE}")
    list(APPEND RIPPLE_RT_NAMES   "${FILE_NAME}")
  endforeach()

  if(RIPPLE_RT_SOURCES STREQUAL "")
    message(WARNING "No sources found for v${HEXAGON_ARCH_VERSION}, skipping.")
    continue()
  endif()

  # Apply the special ripple flag only to HVX_Scatter_Gather
  foreach(SRC IN LISTS RIPPLE_RT_SOURCES)
    get_filename_component(_base "${SRC}" NAME_WE)
    if(_base STREQUAL "HVX_Scatter_Gather")
      set_source_files_properties("${SRC}" PROPERTIES COMPILE_OPTIONS "${RIPPLE_FLAG}")
    endif()
  endforeach()

  # One object library per arch
  add_compiler_rt_object_libraries(
    RippleLib.${HEXAGON_ARCH_VERSION}
    ARCHS   ${HEXAGON}
    SOURCES ${RIPPLE_RT_SOURCES}
    CFLAGS  -O2 ${RIPPLE_FLAG} ${HEXAGON_FLAGS} ${RT_INCLUDES} -emit-llvm
  )

  # Output/Install dirs
  get_compiler_rt_output_dir(${HEXAGON} output_dir)
  get_compiler_rt_install_dir(${HEXAGON} install_dir)
  set(BC_BUILD_DIR ${output_dir}/v${HEXAGON_ARCH_VERSION}/ripple)

  # Create .bc rules per source (no external cmake invocation)
  ripple_add_bc_rules(
    RippleLib.${HEXAGON_ARCH_VERSION}
    ${HEXAGON}
    "${RIPPLE_RT_NAMES}"
    "${BC_BUILD_DIR}"
    BCS_OUT
  )

  add_custom_target(generate_bc_v${HEXAGON_ARCH_VERSION} DEPENDS ${BCS_OUT})
  add_dependencies(RippleLib generate_bc_v${HEXAGON_ARCH_VERSION})

  install(
    FILES ${BCS_OUT}
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DESTINATION ${install_dir}/ripple/v${HEXAGON_ARCH_VERSION}
  )

  message(STATUS "Will emit ripple library for ${HEXAGON}/v${HEXAGON_ARCH_VERSION} to: ${BC_BUILD_DIR}")
endforeach()

# Headers once (not per-arch)
file(GLOB RIPPLE_RT_HEADERS ${RIPPLE_RT_HEADER_DIR}/*.h)
install(
  FILES ${RIPPLE_RT_HEADERS}
  COMPONENT compiler-rt-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ${COMPILER_RT_INSTALL_INCLUDE_DIR}/ripple
)
