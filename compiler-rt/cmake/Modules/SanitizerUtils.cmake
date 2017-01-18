set(SANITIZER_GEN_DYNAMIC_LIST
  ${COMPILER_RT_SOURCE_DIR}/lib/sanitizer_common/scripts/gen_dynamic_list.py)

set(SANITIZER_LINT_SCRIPT
  ${COMPILER_RT_SOURCE_DIR}/lib/sanitizer_common/scripts/check_lint.sh)

# Create a target "<name>-<arch>-symbols" that would generate the list of
# symbols that need to be exported from sanitizer runtime "<name>". Function
# interceptors are exported automatically, user can also provide files with
# symbol names that should be exported as well.
#   add_sanitizer_rt_symbols(<name>
#                            ARCHS <architectures>
#                            PARENT_TARGET <convenience parent target>
#                            EXTRA <files with extra symbols to export>)
macro(add_sanitizer_rt_symbols name)
  cmake_parse_arguments(ARG
    ""
    "PARENT_TARGET"
    "ARCHS;EXTRA"
    ${ARGN})
  foreach(arch ${ARG_ARCHS})
    set(target_name ${name}-${arch})
    set(stamp ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.syms-stamp)
    set(extra_args)
    foreach(arg ${ARG_EXTRA})
      list(APPEND extra_args "--extra" ${arg})
    endforeach()
    add_custom_command(OUTPUT ${stamp}
      COMMAND ${PYTHON_EXECUTABLE}
        ${SANITIZER_GEN_DYNAMIC_LIST} ${extra_args} $<TARGET_FILE:${target_name}>
        > $<TARGET_FILE:${target_name}>.syms
      COMMAND ${CMAKE_COMMAND} -E touch ${stamp}
      DEPENDS ${target_name} ${SANITIZER_GEN_DYNAMIC_LIST} ${ARG_EXTRA}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating exported symbols for ${target_name}"
      VERBATIM)
    add_custom_target(${target_name}-symbols ALL
      DEPENDS ${stamp}
      SOURCES ${SANITIZER_GEN_DYNAMIC_LIST} ${ARG_EXTRA})

    install(FILES $<TARGET_FILE:${target_name}>.syms
            DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
    if(ARG_PARENT_TARGET)
      add_dependencies(${ARG_PARENT_TARGET} ${target_name}-symbols)
    endif()
  endforeach()
endmacro()

# This function is only used on Darwin, where undefined symbols must be specified
# in the linker invocation.
function(add_weak_symbols libname link_flags)
  file(STRINGS "${COMPILER_RT_SOURCE_DIR}/lib/${libname}/weak_symbols.txt" WEAK_SYMBOLS)
  set(local_link_flags ${${link_flags}})
  foreach(SYMBOL ${WEAK_SYMBOLS})
    set(local_link_flags ${local_link_flags} -Wl,-U,${SYMBOL})
  endforeach()
  set(${link_flags} ${local_link_flags} PARENT_SCOPE)
endfunction()

macro(add_sanitizer_rt_version_list name)
  set(vers ${CMAKE_CURRENT_BINARY_DIR}/${name}.vers)
  cmake_parse_arguments(ARG "" "" "LIBS;EXTRA" ${ARGN})
  set(args)
  foreach(arg ${ARG_EXTRA})
    list(APPEND args "--extra" ${arg})
  endforeach()
  foreach(arg ${ARG_LIBS})
    list(APPEND args "$<TARGET_FILE:${arg}>")
  endforeach()
  add_custom_command(OUTPUT ${vers}
    COMMAND ${PYTHON_EXECUTABLE}
      ${SANITIZER_GEN_DYNAMIC_LIST} --version-list ${args}
      > ${vers}
    DEPENDS ${SANITIZER_GEN_DYNAMIC_LIST} ${ARG_EXTRA} ${ARG_LIBS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating version list for ${name}"
    VERBATIM)

  add_custom_target(${name}-version-list ALL
    DEPENDS ${vers})
endmacro()

# Add target to check code style for sanitizer runtimes.
if(CMAKE_HOST_UNIX)
  add_custom_target(SanitizerLintCheck
    COMMAND env LLVM_CHECKOUT=${LLVM_MAIN_SRC_DIR} SILENT=1 TMPDIR=
      PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      COMPILER_RT=${COMPILER_RT_SOURCE_DIR}
      ${SANITIZER_LINT_SCRIPT}
    DEPENDS ${SANITIZER_LINT_SCRIPT}
    COMMENT "Running lint check for sanitizer sources..."
    VERBATIM)
endif()

