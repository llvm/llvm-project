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

target_include_directories(f18
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Create module files directly from the top-level module source directory
foreach(filename ${MODULES})
  set(base ${FLANG_INTRINSIC_MODULES_DIR}/${filename})
  if(${filename} MATCHES "__fortran_type_info")
    set(depends "")
  elseif(${filename} MATCHES "__fortran_builtins")
    set(depends ${FLANG_INTRINSIC_MODULES_DIR}/__fortran_type_info.mod)
  else()
    set(depends ${FLANG_INTRINSIC_MODULES_DIR}/__fortran_builtins.mod)
  endif()
  add_custom_command(OUTPUT ${base}.mod
    COMMAND ${CMAKE_COMMAND} -E make_directory ${FLANG_INTRINSIC_MODULES_DIR}
    COMMAND f18 -fsyntax-only ${FLANG_SOURCE_DIR}/module/${filename}.f90
    WORKING_DIRECTORY ${FLANG_INTRINSIC_MODULES_DIR}
    DEPENDS f18 ${FLANG_SOURCE_DIR}/module/${filename}.f90 ${depends}
  )
  add_custom_command(OUTPUT ${base}.f18.mod
    DEPENDS ${base}.mod
    COMMAND ${CMAKE_COMMAND} -E copy ${base}.mod ${base}.f18.mod)
  list(APPEND MODULE_FILES ${base}.mod ${base}.f18.mod)
  install(FILES ${base}.mod ${base}.f18.mod DESTINATION include/flang)
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
