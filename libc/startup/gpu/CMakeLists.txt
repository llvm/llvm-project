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
    DEPENDS ${ADD_STARTUP_OBJECT_DEPENDS}
    COMPILE_OPTIONS ${ADD_STARTUP_OBJECT_COMPILE_OPTIONS}
  )
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      OUTPUT_NAME ${name}.o
  )
endfunction()

if(LIBC_GPU_TARGET_ARCHITECTURE_IS_AMDGPU)
  add_subdirectory(amdgpu)

  add_startup_object(
    crt1
    ALIAS
    DEPENDS
      .amdgpu.crt1
  )
elseif(LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
  add_subdirectory(nvptx)

  add_startup_object(
    crt1
    ALIAS
    DEPENDS
      .nvptx.crt1
  )
else()
  # Skip building the startup code if there are no supported GPUs.
  message(STATUS "Skipping startup for gpu target, no GPUs were detected")
  return()
endif()

add_custom_target(libc-startup)
set(startup_components crt1)
foreach(target IN LISTS startup_components)
  set(fq_target_name libc.startup.gpu.${target})
  add_dependencies(libc-startup ${fq_target_name})
  install(FILES $<TARGET_OBJECTS:${fq_target_name}>
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RENAME $<TARGET_PROPERTY:${fq_target_name},OUTPUT_NAME>
          COMPONENT libc)
endforeach()
