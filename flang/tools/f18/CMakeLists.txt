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
  "ieee_arithmetic"
  "ieee_exceptions"
  "ieee_features"
  "iso_c_binding"
  "iso_fortran_env"
  "omp_lib"
  "__fortran_builtins"
)

set(include ${FLANG_BINARY_DIR}/include/flang)
target_include_directories(f18
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Create module files directly from the top-level module source directory
foreach(filename ${MODULES})
  if(${filename} MATCHES "__fortran_builtins")
    set(depends "")
  else()
    set(depends ${include}/__fortran_builtins.mod)
  endif()
  add_custom_command(OUTPUT ${include}/${filename}.mod
    COMMAND f18 -fparse-only -I${include}
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

set(FLANG_INTRINSIC_MODULES_DIR ${FLANG_BINARY_DIR}/include/flang)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/flang.sh.in ${CMAKE_BINARY_DIR}/tools/flang/bin/flang @ONLY)
file(COPY ${CMAKE_BINARY_DIR}/tools/flang/bin/flang DESTINATION ${CMAKE_BINARY_DIR}/bin FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)
# The flang script to be installed needs a different path to the headers.
set(FLANG_INTRINSIC_MODULES_DIR ${CMAKE_INSTALL_PREFIX}/include/flang)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/flang.sh.in ${FLANG_BINARY_DIR}/bin/flang-install.sh @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/f18_version.h.in ${CMAKE_CURRENT_BINARY_DIR}/f18_version.h @ONLY)

install(PROGRAMS ${FLANG_BINARY_DIR}/bin/flang-install.sh DESTINATION bin RENAME flang PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)
