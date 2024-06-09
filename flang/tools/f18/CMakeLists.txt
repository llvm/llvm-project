set(LLVM_LINK_COMPONENTS
  FrontendOpenACC
  FrontendOpenMP
  Support
  )

set(MODULES
  "__fortran_builtins"
  "__fortran_ieee_exceptions"
  "__fortran_type_info"
  "__ppc_types"
  "__ppc_intrinsics"
  "mma"
  "__cuda_builtins"
  "cudadevice"
  "ieee_arithmetic"
  "ieee_exceptions"
  "ieee_features"
  "iso_c_binding"
  "iso_fortran_env"
)

# Create module files directly from the top-level module source directory.
# If CMAKE_CROSSCOMPILING, then the newly built flang-new executable was
# cross compiled, and thus can't be executed on the build system and thus
# can't be used for generating module files.
if (NOT CMAKE_CROSSCOMPILING)
  foreach(filename ${MODULES})
    set(depends "")
    set(opts "")
    if(${filename} STREQUAL "__fortran_builtins" OR
       ${filename} STREQUAL "__ppc_types")
    elseif(${filename} STREQUAL "__ppc_intrinsics" OR
           ${filename} STREQUAL "mma")
      set(depends ${FLANG_INTRINSIC_MODULES_DIR}/__ppc_types.mod)
    elseif(${filename} STREQUAL "cudadevice")
      set(opts -fc1 -xcuda)
      set(depends ${FLANG_INTRINSIC_MODULES_DIR}/__cuda_builtins.mod)
    else()
      set(depends ${FLANG_INTRINSIC_MODULES_DIR}/__fortran_builtins.mod)
      if(NOT ${filename} STREQUAL "__fortran_type_info")
        set(depends ${depends} ${FLANG_INTRINSIC_MODULES_DIR}/__fortran_type_info.mod)
      endif()
      if(${filename} STREQUAL "ieee_arithmetic" OR
         ${filename} STREQUAL "ieee_exceptions")
        set(depends ${depends} ${FLANG_INTRINSIC_MODULES_DIR}/__fortran_ieee_exceptions.mod)
      endif()
    endif()

    # The module contains PPC vector types that needs the PPC target.
    if(${filename} STREQUAL "__ppc_intrinsics" OR
       ${filename} STREQUAL "mma")
      if (PowerPC IN_LIST LLVM_TARGETS_TO_BUILD)
        set(opts "--target=ppc64le")
      else()
        # Do not compile PPC module if the target is not available.
        continue()
      endif()
    endif()

    set(base ${FLANG_INTRINSIC_MODULES_DIR}/${filename})
    # TODO: We may need to flag this with conditional, in case Flang is built w/o OpenMP support
    add_custom_command(OUTPUT ${base}.mod
      COMMAND ${CMAKE_COMMAND} -E make_directory ${FLANG_INTRINSIC_MODULES_DIR}
      COMMAND flang-new ${opts} -cpp -fsyntax-only -module-dir ${FLANG_INTRINSIC_MODULES_DIR}
        ${FLANG_SOURCE_DIR}/module/${filename}.f90
      DEPENDS flang-new ${FLANG_SOURCE_DIR}/module/${filename}.f90 ${FLANG_SOURCE_DIR}/module/__fortran_builtins.f90 ${depends}
    )
    list(APPEND MODULE_FILES ${base}.mod)
    install(FILES ${base}.mod DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/flang")
  endforeach()

  # Special case for omp_lib.mod, because its source comes from openmp/runtime/src/include.
  # It also produces two module files: omp_lib.mod and omp_lib_kinds.mod.  Compile these
  # files only if OpenMP support has been configured.
  if (LLVM_TOOL_OPENMP_BUILD)
    message(STATUS "OpenMP runtime support enabled via LLVM_ENABLED_PROJECTS, building omp_lib.mod")
    set(base ${FLANG_INTRINSIC_MODULES_DIR}/omp_lib)
    add_custom_command(OUTPUT ${base}.mod ${base}_kinds.mod
      COMMAND ${CMAKE_COMMAND} -E make_directory ${FLANG_INTRINSIC_MODULES_DIR}
      COMMAND flang-new -cpp -fsyntax-only ${opts} -module-dir ${FLANG_INTRINSIC_MODULES_DIR}
        ${CMAKE_BINARY_DIR}/projects/openmp/runtime/src/omp_lib.F90
      DEPENDS flang-new ${FLANG_INTRINSIC_MODULES_DIR}/iso_c_binding.mod ${CMAKE_BINARY_DIR}/projects/openmp/runtime/src/omp_lib.F90 ${depends}
    )
    add_custom_command(OUTPUT ${base}.f18.mod
      DEPENDS ${base}.mod
      COMMAND ${CMAKE_COMMAND} -E copy ${base}.mod ${base}.f18.mod)
      add_custom_command(OUTPUT ${base}_kinds.f18.mod
        DEPENDS ${base}.mod
        COMMAND ${CMAKE_COMMAND} -E copy ${base}_kinds.mod ${base}_kinds.f18.mod)
    list(APPEND MODULE_FILES ${base}.mod ${base}.f18.mod ${base}_kinds.mod ${base}_kinds.f18.mod)
    install(FILES ${base}.mod ${base}.f18.mod ${base}_kinds.mod ${base}_kinds.f18.mod DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/flang")
  else()
    message(STATUS "Not building omp_lib.mod, no OpenMP runtime in LLVM_ENABLED_PROJECTS")
  endif()
endif()

add_custom_target(module_files ALL DEPENDS ${MODULE_FILES})
set_target_properties(module_files PROPERTIES FOLDER "Flang/Resources")

# TODO Move this to a more suitable location
# Copy the generated omp_lib.h header file, if OpenMP support has been configured.
if (LLVM_TOOL_OPENMP_BUILD)
  message(STATUS "OpenMP runtime support enabled via LLVM_ENABLED_PROJECTS, building omp_lib.h")
  file(COPY ${CMAKE_BINARY_DIR}/projects/openmp/runtime/src/omp_lib.h DESTINATION "${CMAKE_BINARY_DIR}/include/flang/OpenMP/" FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
  install(FILES ${CMAKE_BINARY_DIR}/include/flang/OpenMP/omp_lib.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/flang/OpenMP")
else()
  message(STATUS "Not copying omp_lib.h, no OpenMP runtime in LLVM_ENABLED_PROJECTS")
endif()
