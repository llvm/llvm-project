# This function merges multiple objects into a single relocatable object
#                     cc -r obj1.o obj2.o -o obj.o
# A relocatable object is an object file that is not fully linked into an
# executable or a shared library. It is an intermediate file format that can
# be passed into the linker.
# A crt object has arch-specific code and arch-agnostic code. To reduce code
# cohesion, the implementation is splitted into multiple units. As a result,
# we need to merge them into a single relocatable object.
function(merge_relocatable_object name)
  set(obj_list "")
  set(fq_link_libraries "")
  get_fq_deps_list(fq_dep_list ${ARGN})
  foreach(target IN LISTS fq_dep_list)
    list(APPEND obj_list "$<TARGET_OBJECTS:${target}>")
    get_target_property(libs ${target} DEPS)
    list(APPEND fq_link_libraries "${libs}")
  endforeach()
  get_fq_target_name(${name} fq_name)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${name}.o
    COMMAND ${CMAKE_CXX_COMPILER} -r ${obj_list} -o ${CMAKE_CURRENT_BINARY_DIR}/${name}.o
    DEPENDS ${obj_list}
    COMMAND_EXPAND_LISTS
  )
  add_custom_target(${fq_name}.relocatable DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${name}.o)
  add_library(${fq_name} OBJECT IMPORTED GLOBAL)
  add_dependencies(${fq_name} ${fq_name}.relocatable)
  target_link_libraries(${fq_name} INTERFACE ${fq_link_libraries})
  set_target_properties(
    ${fq_name} 
    PROPERTIES
      LINKER_LANGUAGE CXX
      OBJECT_FILES ${CMAKE_CURRENT_BINARY_DIR}/${name}.o
      IMPORTED_OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/${name}.o
      TARGET_TYPE ${OBJECT_LIBRARY_TARGET_TYPE}
      DEPS "${fq_link_libraries}"
  ) 
endfunction()

function(add_startup_object name)
  cmake_parse_arguments(
    "ADD_STARTUP_OBJECT"
    "" # Option argument
    "SRC"   # Single value arguments
    "DEPENDS;COMPILE_OPTIONS" # Multi value arguments
    ${ARGN}
  )

  get_fq_target_name(${name} fq_target_name)
  
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

if(NOT (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE}))
  message(STATUS "Skipping startup for target architecture ${LIBC_TARGET_ARCHITECTURE}")
  return()
endif()

add_subdirectory(${LIBC_TARGET_ARCHITECTURE})

# TODO: factor out crt1 into multiple objects
merge_relocatable_object(
  crt1
  .${LIBC_TARGET_ARCHITECTURE}.crt1
)

add_startup_object(
  crti
  SRC
    crti.cpp
)

add_startup_object(
  crtn
  SRC
    crtn.cpp
)

add_custom_target(libc-startup)
set(startup_components crt1 crti crtn)
foreach(target IN LISTS startup_components)
  set(fq_target_name libc.startup.linux.${target})
  add_dependencies(libc-startup ${fq_target_name})
  install(FILES $<TARGET_OBJECTS:${fq_target_name}>
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RENAME $<TARGET_PROPERTY:${fq_target_name},OUTPUT_NAME>
          COMPONENT libc)
endforeach()
