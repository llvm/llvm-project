#===-- tools/f18/CMakeLists.txt --------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

add_executable(f18
  f18.cpp
  dump.cpp
)

target_link_libraries(f18
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranLower
  LLVMSupport
)

add_executable(f18-parse-demo
  f18-parse-demo.cpp
  stub-evaluate.cpp
)

target_link_libraries(f18-parse-demo
  FortranParser
)

set_target_properties(f18 f18-parse-demo
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

set(MODULES
  "ieee_arithmetic"
  "ieee_exceptions"
  "ieee_features"
  "iso_c_binding"
  "iso_fortran_env"
  "omp_lib"
  "__fortran_builtins"
)

set(include ${CMAKE_CURRENT_BINARY_DIR}/include)

# Create module files directly from the top-level module source directory
foreach(filename ${MODULES})
  if(${filename} MATCHES "__fortran_builtins")
    set(depends "")
  else()
    set(depends ${include}/__fortran_builtins.mod)
  endif()
  add_custom_command(OUTPUT ${include}/${filename}.mod
    COMMAND f18 -fparse-only -I${include}
      ${PROJECT_SOURCE_DIR}/module/${filename}.f90
    WORKING_DIRECTORY ${include}
    DEPENDS f18 ${PROJECT_SOURCE_DIR}/module/${filename}.f90 ${depends}
  )
  add_custom_command(OUTPUT ${include}/${filename}.f18.mod
    DEPENDS ${include}/${filename}.mod
    COMMAND ${CMAKE_COMMAND} -E
      copy ${include}/${filename}.mod ${include}/${filename}.f18.mod)
  list(APPEND MODULE_FILES ${include}/${filename}.mod)
  list(APPEND MODULE_FILES ${include}/${filename}.f18.mod)
  install(FILES ${include}/${filename}.mod DESTINATION include)
  install(FILES ${include}/${filename}.f18.mod DESTINATION include)
endforeach()

add_custom_target(module_files ALL DEPENDS ${MODULE_FILES})

install(TARGETS f18 f18-parse-demo DESTINATION bin)

file(COPY flang.sh
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/bin"
  FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)
file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/bin/flang.sh" "${CMAKE_CURRENT_BINARY_DIR}/bin/flang")

install(PROGRAMS flang.sh DESTINATION bin RENAME flang)
