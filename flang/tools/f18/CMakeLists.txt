set(LLVM_LINK_COMPONENTS
  FrontendOpenACC
  FrontendOpenMP
  Support
  )
add_flang_tool(f18
  dump.cpp
  f18.cpp
  )

target_link_libraries(f18
  PRIVATE
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranLower
)

set(MODULES
  "__fortran_builtins"
  "__fortran_type_info"
  "ieee_arithmetic"
  "ieee_exceptions"
  "ieee_features"
  "iso_c_binding"
  "iso_fortran_env"
  "omp_lib"
  "__fortran_builtins"
  "__fortran_type_info"
)

set(include ${FLANG_BINARY_DIR}/include/flang)
target_include_directories(f18
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Create module files directly from the top-level module source directory
foreach(filename ${MODULES})
  if(${filename} MATCHES "__fortran_type_info")
    set(depends "")
  elseif(${filename} MATCHES "__fortran_builtins")
    set(depends ${include}/__fortran_type_info.mod)
  else()
    set(depends ${include}/__fortran_builtins.mod)
  endif()
  add_custom_command(OUTPUT ${include}/${filename}.mod
    COMMAND f18 -fsyntax-only -I${include}
      ${FLANG_SOURCE_DIR}/module/${filename}.f90
    WORKING_DIRECTORY ${include}
    DEPENDS f18 ${FLANG_SOURCE_DIR}/module/${filename}.f90 ${depends}
  )
  add_custom_command(OUTPUT ${include}/${filename}.f18.mod
    DEPENDS ${include}/${filename}.mod
    COMMAND ${CMAKE_COMMAND} -E
      copy ${include}/${filename}.mod ${include}/${filename}.f18.mod)
  list(APPEND MODULE_FILES ${include}/${filename}.mod)
  list(APPEND MODULE_FILES ${include}/${filename}.f18.mod)
  install(FILES ${include}/${filename}.mod DESTINATION include/flang)
  install(FILES ${include}/${filename}.f18.mod DESTINATION include/flang)
endforeach()

add_custom_target(module_files ALL DEPENDS ${MODULE_FILES})

install(TARGETS f18 DESTINATION bin)

# This flang shell script will only work in a POSIX shell.
if (NOT WIN32)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/flang
    DESTINATION ${CMAKE_BINARY_DIR}/bin
    FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/flang DESTINATION bin)
endif()
