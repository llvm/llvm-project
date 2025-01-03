#===-- cmake/modules/AddFlangRT.cmake --------------------------------------===#
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

  if (ARG_INSTALL_WITH_TOOLCHAIN AND ARG_EXCLUDE_FROM_ALL)
    message(SEND_ERROR "add_flangrt_library(${name} ...):
        INSTALL_WITH_TOOLCHAIN and EXCLUDE_FROM_ALL are in conflict. When
        installing an artifact it must have been built first in the 'all' target.
      ")
  endif ()

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
  if (ARG_EXCLUDE_FROM_ALL)
    list(APPEND extra_args EXCLUDE_FROM_ALL)
  endif ()

  # Also add header files to IDEs to list as part of the library.
  set_source_files_properties(${ARG_ADDITIONAL_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)

  add_library(${name} ${extra_args} ${ARG_ADDITIONAL_HEADERS} ${ARG_UNPARSED_ARGUMENTS})

  if (ARG_INSTALL_WITH_TOOLCHAIN)
    set_target_properties(${name} PROPERTIES FOLDER "Flang-RT/Toolchain Libraries")
  elseif (ARG_OBJECT)
    set_target_properties(${name} PROPERTIES FOLDER "Flang-RT/Object Libraries")
  else ()
    set_target_properties(${name} PROPERTIES FOLDER "Flang-RT/Libraries")
  endif ()

  # Minimum required C++ version for Flang-RT, even if CMAKE_CXX_STANDARD is defined to something else.
  target_compile_features(${name} PRIVATE cxx_std_17)

  # Use compiler-specific options to disable exceptions and RTTI.
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti -fno-unwind-tables -fno-asynchronous-unwind-tables>
      )
  elseif (MSVC)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/EHs-c- /GR->
      )
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "XL")
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-qnoeh -qnortti>
      )
  endif ()

  # Also for CUDA source when compiling with FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT=CUDA
  if (CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    # Assuming gcc as host compiler.
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions -Xcompiler -fno-rtti -Xcompiler -fno-unwind-tables -Xcompiler -fno-asynchronous-unwind-tables>
      )
  else ()
    # Assuming a clang-compatible CUDA compiler.
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-fno-exceptions -fno-rtti -fno-unwind-tables -fno-asynchronous-unwind-tables>
      )
  endif ()

  # Flang-RT's public headers
  target_include_directories(${name} PRIVATE "${FLANG_RT_SOURCE_DIR}/include")

  # For ISO_Fortran_binding.h to be found by the runtime itself (Accessed as #include "flang/ISO_Fortran_binding.h")
  # User applications can use #include <ISO_Fortran_binding.h>
  target_include_directories(${name} PRIVATE "${FLANG_SOURCE_DIR}/include")

  # For Flang-RT's configured config.h to be found
  target_include_directories(${name} PRIVATE "${FLANG_RT_BINARY_DIR}")

  # Disable libstdc++/libc++ assertions, even in an LLVM_ENABLE_ASSERTIONS
  # build, to avoid an unwanted dependency on libstdc++/libc++.so.
  if (FLANG_RT_SUPPORTS_UNDEFINE_FLAG)
    target_compile_options(${name} PUBLIC -U_GLIBCXX_ASSERTIONS)
    target_compile_options(${name} PUBLIC -U_LIBCPP_ENABLE_ASSERTIONS)
  endif ()

  # Flang/Clang (including clang-cl) -compiled programs targeting the MSVC ABI
  # should only depend on msvcrt/ucrt. LLVM still emits libgcc/compiler-rt
  # functions in some cases like 128-bit integer math (__udivti3, __modti3,
  # __fixsfti, __floattidf, ...) that msvc does not support. We are injecting a
  # dependency to Compiler-RT's builtin library where these are implemented.
  if (MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if (FLANG_RT_BUILTINS_LIBRARY)
      target_compile_options(${name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:-Xclang>" "$<$<COMPILE_LANGUAGE:CXX,C>:--dependent-lib=${FLANG_RT_BUILTINS_LIBRARY}>")
    endif ()
  endif ()
  if (MSVC AND CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
    if (FLANG_RT_BUILTINS_LIBRARY)
      target_compile_options(${name} PRIVATE "$<$<COMPILE_LANGUAGE:Fortran>:-Xflang>" "$<$<COMPILE_LANGUAGE:Fortran>:--dependent-lib=${FLANG_RT_BUILTINS_LIBRARY}>")
    else ()
      message(WARNING "Did not find libclang_rt.builtins.lib.
        LLVM may emit builtins that are not implemented in msvcrt/ucrt and
        instead falls back to builtins from Compiler-RT. Linking with ${name}
        may result in a linker error.")
    endif ()
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
        ARCHIVE_OUTPUT_DIRECTORY "${FLANG_RT_OUTPUT_RESOURCE_LIB_DIR}"
      )

    install(TARGETS ${name}
        ARCHIVE DESTINATION "${FLANG_RT_INSTALL_RESOURCE_LIB_PATH}"
      )
  endif ()

  # flang-rt should build all the Flang-RT targets that are built in an
  # 'all' build.
  if (NOT ARG_EXCLUDE_FROM_ALL)
    add_dependencies(flang-rt ${name})
  endif ()
endfunction (add_flangrt_library)
