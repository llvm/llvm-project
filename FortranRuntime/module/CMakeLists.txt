#===-- module/CMakeLists.txt -----------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

set(module_sources
  "__cuda_builtins.f90"
  "__fortran_builtins.f90"
  "__fortran_ieee_exceptions.f90"
  "__fortran_type_info.f90"
  # "cudadevice.f90" # TODO: -fc1 -xcuda
  "ieee_arithmetic.f90"
  "ieee_exceptions.f90"
  "ieee_features.f90"
  "iso_c_binding.f90"
  "iso_fortran_env.f90"
)

set_source_files_properties(
 "iso_fortran_env.f90"
 PROPERTIES
   COMPILE_FLAGS "-mmlir -ignore-module-only-builtins"
)

set_source_files_properties(
  "__fortran_builtins.f90"
   PROPERTIES
     COMPILE_FLAGS "-mmlir -ignore-missing-type-desc"
)

# Requires PowerPC vector instructions
if (LLVM_TARGET_TRIPLE MATCHES "(ppc|powerpc)(32|64)?(le|be)?\-.*")
  list(APPEND module_sources
    "__ppc_types.f90"
    "__ppc_intrinsics.f90"
    "mma.f90"
  )
endif ()

add_fortranruntime_library(module_files OBJECT
  ${module_sources}
)

# Usually, CMake will ignore dependencies from Fortran intrinsic modules. Since this is building those intrinsic modules themselves, tell CMake to not prune such dependencies.
set_target_properties(module_files
  PROPERTIES
    Fortran_BUILDING_INSTRINSIC_MODULES ON
)

# The usual technique to install files in CMake is
#
#   install(TARGET <targetname> <artifacttype> ...)
#
# However, there is no <artifacttype> for Fortran modules. We have to derive the module file path individually.
set(module_files "")
foreach (f90file IN LISTS module_sources)
   get_filename_component(base "${f90file}" NAME_WE )
   list(APPEND module_files "${FORTRANRUNTIME_BUILD_INCLUDE_DIR}/flang/${base}.mod")
endforeach ()
install(FILES ${module_files} DESTINATION "${FORTRANRUNTIME_INSTALL_INCLUDE_DIR}/flang")
