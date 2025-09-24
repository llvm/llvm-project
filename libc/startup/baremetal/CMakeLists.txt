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

add_entrypoint_object(
  init
  SRCS
    init.cpp
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.common
  HDRS
    init.h
)

add_entrypoint_object(
  fini
  SRCS
    fini.cpp
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.common
  HDRS
    fini.h
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
else()
  message(WARNING "Cannot build 'crt1.o' for ${LIBC_TARGET_ARCHITECTURE} yet.")
  return()
endif()

add_startup_object(
  crt1
  ALIAS
  DEPENDS
  .${LIBC_TARGET_ARCHITECTURE}.crt1
)

add_custom_target(libc-startup)

set(fq_target_name libc.startup.baremetal.${LIBC_TARGET_ARCHITECTURE}.crt1)
add_dependencies(libc-startup ${fq_target_name})
install(FILES $<TARGET_OBJECTS:${fq_target_name}>
        DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
        RENAME $<TARGET_PROPERTY:${fq_target_name},OUTPUT_NAME>
        COMPONENT libc)
