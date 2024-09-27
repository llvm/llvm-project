#===-- cmake/modules/AddFortranRuntime.cmake -------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# Builds a library with common option for FortranRuntime.
#
# Usage:
#
# add_fortranruntime_library(name sources ...
#   SHARED
#     Build a dynamic (.so/.dll) library
#   STATIC
#     Build a static (.a/.lib) library
#   OBJECT
#     Create only object- and Fortran module files without static/dynamic library
#   INSTALL_WITH_TOOLCHAIN
#     Install library into Clang's resource directory so it can be found by the Flang driver during compilation, including tests
#   EXCLUDE_FROM_ALL
#     Do not build library by default; typically used for libraries needed for testing only, no install
#   LINK_TO_LLVM
#     Library requires include path and linking to LLVM's Support component
#   ADDITIONAL_HEADERS
#     May specify header files for IDE generators.
# )
function (add_fortranruntime_library name)
  set(options STATIC SHARED OBJECT INSTALL_WITH_TOOLCHAIN EXCLUDE_FROM_ALL LINK_TO_LLVM)
  set(multiValueArgs ADDITIONAL_HEADERS)
  cmake_parse_arguments(ARG
    "${options}"
    ""
    "${multiValueArgs}"
    ${ARGN})

  # Also add header files to IDEs to list as part of the library
  set_source_files_properties(${ARG_ADDITIONAL_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)

  # Forward libtype to add_library
  set(extra_args "")
  if (ARG_SHARED)
    list(APPEND extra_args SHARED)
  endif ()
  if (ARG_STATIC)
    list(APPEND extra_args STATIC)
  endif ()
  if (ARG_OBJECT)
    list(APPEND extra_args OBJECT)
  endif ()
  if (EXCLUDE_FROM_ALL)
    list(APPEND extra_args EXCLUDE_FROM_ALL)
  endif ()

  add_library(${name} ${extra_args} ${ARG_ADDITIONAL_HEADERS} ${ARG_UNPARSED_ARGUMENTS}   )
  if (ARG_INSTALL_WITH_TOOLCHAIN)
    set_target_properties(${name} PROPERTIES FOLDER "Fortran Runtime/Toolchain Libraries")
  elseif (ARG_OBJECT)
    set_target_properties(${name} PROPERTIES FOLDER "Fortran Runtime/Object Libraries")
  else ()
    set_target_properties(${name} PROPERTIES FOLDER "Fortran Runtime/Libraries")
  endif ()

  target_compile_features(${name} PRIVATE cxx_std_17)
     if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
       target_compile_options (${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti -fno-unwind-tables -fno-asynchronous-unwind-tables>)
    elseif(MSVC)
      target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/EHs-c- /GR->)
    endif()

  target_include_directories(${name} PRIVATE "${FORTRANRUNTIME_BINARY_DIR}")   # For configured config.h for be found
  target_include_directories(${name} PRIVATE "${FORTRANRUNTIME_SOURCE_DIR}/include")

set_target_properties(${name}
  PROPERTIES
    Fortran_MODULE_DIRECTORY "${FORTRANRUNTIME_BUILD_INCLUDE_DIR}/flang"
)

  if (ARG_LINK_TO_LLVM)
    if (LLVM_LINK_LLVM_DYLIB)
      set(llvm_libs LLVM)
    else()
      llvm_map_components_to_libnames(llvm_libs Support)
    endif()
    target_link_libraries(${name} PUBLIC  ${llvm_libs})
     target_include_directories(${name} PRIVATE  ${LLVM_INCLUDE_DIRS})
  endif ()

  # If this is part of the toolchain, put it into the compiler's resource directory.
  # Otherwise it is part of testing and is not installed at all.
  # TODO: Consider multi-configuration builds
  if (INSTALL_WITH_TOOLCHAIN)
    set_target_properties(${name}
      PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY "${FORTRANRUNTIME_BUILD_LIB_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${FORTRANRUNTIME_BUILD_LIB_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${FORTRANRUNTIME_BUILD_LIB_DIR}"
    )

    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
      install(TARGETS ${name}
        LIBRARY DESTINATION "${FORTRANRUNTIME_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${FORTRANRUNTIME_INSTALL_LIB_DIR}"
        RUNTIME DESTINATION "${FORTRANRUNTIME_INSTALL_LIB_DIR}"
      )
    endif ()
  endif ()
endfunction (add_fortranruntime_library)
