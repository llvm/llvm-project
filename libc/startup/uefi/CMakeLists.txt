# TODO: Use generic "add_startup_object" https://github.com/llvm/llvm-project/issues/133156
function(add_startup_object name)
  cmake_parse_arguments(
    "ADD_STARTUP_OBJECT"
    "ALIAS" # Option argument
    "SRC"   # Single value arguments
    "DEPENDS;COMPILE_OPTIONS" # Multi value arguments
    ${ARGN}
  )

  get_fq_target_name(${name} fq_target_name)
  if(ADD_STARTUP_OBJECT_ALIAS)
    get_fq_deps_list(fq_dep_list ${ADD_STARTUP_OBJECT_DEPENDS})
    add_library(${fq_target_name} ALIAS ${fq_dep_list})
    return()
  endif()

  add_object_library(
    ${name}
    SRCS ${ADD_STARTUP_OBJECT_SRC}
    COMPILE_OPTIONS ${ADD_STARTUP_OBJECT_COMPILE_OPTIONS}
    ${ADD_STARTUP_OBJECT_UNPARSED_ARGUMENTS}
    DEPENDS ${ADD_STARTUP_OBJECT_DEPENDS}
  )
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      OUTPUT_NAME ${name}.o
  )
endfunction()

add_startup_object(
  crt1
  SRCS
    crt1.cpp
  DEPENDS
    libc.config.app_h
    libc.src.__support.OSUtil.uefi.uefi_util
)

add_custom_target(libc-startup)
set(startup_components crt1)
foreach(target IN LISTS startup_components)
  set(fq_target_name libc.startup.uefi.${target})
  add_dependencies(libc-startup ${fq_target_name})
  install(FILES $<TARGET_OBJECTS:${fq_target_name}>
          DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
          RENAME $<TARGET_PROPERTY:${fq_target_name},OUTPUT_NAME>
          COMPONENT libc)
endforeach()
