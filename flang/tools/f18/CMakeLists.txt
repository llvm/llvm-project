set(LLVM_LINK_COMPONENTS
  FrontendOpenACC
  FrontendOpenMP
  Support
  )

set(MODULES
  "__fortran_builtins"
  "__fortran_ieee_exceptions"
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

# Create module files directly from the top-level module source directory
foreach(filename ${MODULES})
  set(base ${FLANG_INTRINSIC_MODULES_DIR}/${filename})
  if(${filename} STREQUAL "__fortran_builtins")
    set(depends "")
  else()
    set(depends ${FLANG_INTRINSIC_MODULES_DIR}/__fortran_builtins.mod)
    if(NOT ${filename} STREQUAL "__fortran_type_info")
      set(depends ${FLANG_INTRINSIC_MODULES_DIR}/__fortran_type_info.mod)
    endif()
    if(${filename} STREQUAL "ieee_arithmetic" OR
       ${filename} STREQUAL "ieee_exceptions")
      set(depends ${FLANG_INTRINSIC_MODULES_DIR}/__fortran_ieee_exceptions.mod)
    endif()
  endif()
  add_custom_command(OUTPUT ${base}.mod
    COMMAND ${CMAKE_COMMAND} -E make_directory ${FLANG_INTRINSIC_MODULES_DIR}
    COMMAND flang-new -fc1 -fsyntax-only -module-dir ${FLANG_INTRINSIC_MODULES_DIR}
      ${FLANG_SOURCE_DIR}/module/${filename}.f90
    DEPENDS flang-new ${FLANG_SOURCE_DIR}/module/${filename}.f90 ${depends}
  )
  add_custom_command(OUTPUT ${base}.f18.mod
    DEPENDS ${base}.mod
    COMMAND ${CMAKE_COMMAND} -E copy ${base}.mod ${base}.f18.mod)
  list(APPEND MODULE_FILES ${base}.mod ${base}.f18.mod)
  install(FILES ${base}.mod ${base}.f18.mod DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/flang")
endforeach()

add_custom_target(module_files ALL DEPENDS ${MODULE_FILES})

# This flang shell script will only work in a POSIX shell.
if (NOT WIN32)
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/bin/flang-to-external-fc
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/flang-to-external-fc
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/flang-to-external-fc ${CMAKE_BINARY_DIR}/bin)
  add_custom_target(flang-to-external-fc ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/flang-to-external-fc)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/flang-to-external-fc DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
add_custom_target(flang-slink ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/flang-to-external-fc COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_BINARY_DIR}/bin/flang-to-external-fc ${CMAKE_BINARY_DIR}/bin/flang)

# TODO Move this to a more suitable location
file(COPY ${FLANG_SOURCE_DIR}/module/omp_lib.h DESTINATION "${CMAKE_BINARY_DIR}/include/flang/OpenMP/" FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
