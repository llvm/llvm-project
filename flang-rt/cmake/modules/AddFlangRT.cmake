#===-- cmake/modules/AddFlangRT.cmake -------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# Builds a library with common options for flang-rt.
#
# Usage:
#
# add_flangrt_library(name sources ...
#   SHARED
#     Build a dynamic (.so/.dll) library
#   STATIC
#     Build a static (.a/.lib) library
#   OBJECT
#     Create only object files without static/dynamic library
#   INSTALL_WITH_TOOLCHAIN
#     Install library into Clang's resource directory so it can be found by the
#     Flang driver during compilation, including tests
#   EXCLUDE_FROM_ALL
#     Do not build library by default; typically used for libraries needed for
#     testing only, no install
#   LINK_TO_LLVM
#     Library requires include path and linking to LLVM's Support component
#   ADDITIONAL_HEADERS
#     May specify header files for IDE generators.
# )
function (add_flangrt_library name)
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

  add_library(${name} ${extra_args} ${ARG_ADDITIONAL_HEADERS} ${ARG_UNPARSED_ARGUMENTS})

  if (ARG_INSTALL_WITH_TOOLCHAIN)
    set_target_properties(${name} PROPERTIES FOLDER "Fortran Runtime/Toolchain Libraries")
  elseif (ARG_OBJECT)
    set_target_properties(${name} PROPERTIES FOLDER "Fortran Runtime/Object Libraries")
  else ()
    set_target_properties(${name} PROPERTIES FOLDER "Fortran Runtime/Libraries")
  endif ()

  target_compile_features(${name} PRIVATE cxx_std_17)
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti -fno-unwind-tables -fno-asynchronous-unwind-tables>
      )
  elseif (MSVC)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/EHs-c- /GR->
      )
  endif ()

  # Flang-rt's public headers
  target_include_directories(${name} PRIVATE "${FLANGRT_SOURCE_DIR}/include")

  # For ISO_Fortran_binding.h to be found by the runtime itself (Accessed as #include "flang/ISO_Fortran_binding.h")
  # User applications can use #include <ISO_Fortran_binding.h>
  target_include_directories(${name} PRIVATE "${FLANG_SOURCE_DIR}/include")

  # For flang-rt's configured config.h to be found
  target_include_directories(${name} PRIVATE "${FLANGRT_BINARY_DIR}")

  # Flang/Clang (including clang-cl) -compiled programs targeting the MSVC ABI
  # should only depend on msv(u)crt. LLVM still emits libgcc/compiler-rt
  # functions for 128-bit integer math (__udivti3, __modti3, __fixsfti,
  # __floattidf, ...) that msvc does not support. We are injecting a dependency
  # to Compiler-RT where these are implemented.
  if (MSVC AND (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang") AND FLANGRT_LIBCALL)
    target_compile_options(${name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:-Xclang>$<$<COMPILE_LANGUAGE:Fortran>:-Xflang>" "--dependent-lib=${FLANGRT_LIBCALL}")
  endif ()

  # Non-GTest unittests depend on LLVMSupport
  if (ARG_LINK_TO_LLVM)
    if (LLVM_LINK_LLVM_DYLIB)
      set(llvm_libs LLVM)
    else()
      llvm_map_components_to_libnames(llvm_libs Support)
    endif()
    target_link_libraries(${name} PUBLIC ${llvm_libs})
    target_include_directories(${name} PUBLIC ${LLVM_INCLUDE_DIRS})
  endif ()

  # If this is part of the toolchain, put it into the compiler's resource
  # directory. Otherwise it is part of testing and is not installed at all.
  # TODO: Consider multi-configuration builds (MSVC_IDE, "Ninja Multi-Config")
  if (ARG_INSTALL_WITH_TOOLCHAIN)
    set_target_properties(${name}
      PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${FLANGRT_BUILD_LIB_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${FLANGRT_BUILD_LIB_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${FLANGRT_BUILD_LIB_DIR}"
      )

    install(TARGETS ${name}
        LIBRARY DESTINATION "${FLANGRT_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${FLANGRT_INSTALL_LIB_DIR}"
        RUNTIME DESTINATION "${FLANGRT_INSTALL_LIB_DIR}"
      )
  endif ()
endfunction (add_flangrt_library)
