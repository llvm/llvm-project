#===-- lib/flang_rt/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

include(AddFlangRTOffload)
# function checks
find_package(Backtrace)
set(HAVE_BACKTRACE ${Backtrace_FOUND})
set(BACKTRACE_HEADER ${Backtrace_HEADER})


# List of files that are buildable for all devices.
set(supported_sources
  ISO_Fortran_binding.cpp
  allocator-registry.cpp
  allocatable.cpp
  array-constructor.cpp
  assign.cpp
  buffer.cpp
  character.cpp
  connection.cpp
  copy.cpp
  derived-api.cpp
  derived.cpp
  descriptor-io.cpp
  descriptor.cpp
  dot-product.cpp
  edit-input.cpp
  edit-output.cpp
  environment.cpp
  external-unit.cpp
  extrema.cpp
  file.cpp
  findloc.cpp
  format.cpp
  inquiry.cpp
  internal-unit.cpp
  io-api.cpp
  io-api-minimal.cpp
  io-error.cpp
  io-stmt.cpp
  iostat.cpp
  matmul-transpose.cpp
  matmul.cpp
  memory.cpp
  misc-intrinsic.cpp
  namelist.cpp
  non-tbp-dio.cpp
  numeric.cpp
  pointer.cpp
  product.cpp
  pseudo-unit.cpp
  ragged.cpp
  stat.cpp
  sum.cpp
  support.cpp
  terminator.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  type-info.cpp
  unit.cpp
  unit-map.cpp
  utf.cpp
)

# List of source not used for GPU offloading.
set(host_sources
  ${FLANG_SOURCE_DIR}/lib/Common/binary-to-decimal.cpp
  ${FLANG_SOURCE_DIR}/lib/Common/decimal-to-binary.cpp
  command.cpp
  complex-powi.cpp
  complex-reduction.c
  exceptions.cpp
  execute.cpp
  extensions.cpp
  ${FLANG_SOURCE_DIR}/module/iso_fortran_env_impl.f90
  main.cpp
  random.cpp
  reduce.cpp
  reduction.cpp
  stop.cpp
  temporary-stack.cpp
  time-intrinsic.cpp
)

file(GLOB_RECURSE public_headers
  "${FLANG_RT_SOURCE_DIR}/include/flang_rt/*.h"
  "${FLANG_SOURCE_DIR}/include/flang/Common/*.h"
  )

file(GLOB_RECURSE private_headers
  "${FLANG_RT_SOURCE_DIR}/lib/flang_rt/*.h"
  "${FLANG_SOURCE_DIR}/lib/Common/*.h"
  )


# Import changes from sibling FortranFloat128Math
get_target_property(f128_sources
  FortranFloat128MathILib INTERFACE_SOURCES
  )
if (f128_sources)
  # The interface may define special macros for Float128Math files,
  # so we need to propagate them.
  get_target_property(f128_defs
    FortranFloat128MathILib INTERFACE_COMPILE_DEFINITIONS
    )
  set_property(SOURCE ${f128_sources}
    APPEND PROPERTY COMPILE_DEFINITIONS
    ${f128_defs}
    )
  get_target_property(f128_include_dirs
    FortranFloat128MathILib INTERFACE_INCLUDE_DIRECTORIES
    )
  set_property(SOURCE ${f128_sources}
    APPEND PROPERTY INCLUDE_DIRECTORIES
    ${f128_include_dirs}
    )
else ()
  set(f128_sources "")
endif ()


set(sources ${supported_sources} ${host_sources} ${f128_sources})

if (NOT WIN32)
  add_flangrt_library(flang_rt STATIC SHARED
    ${sources}
    INSTALL_WITH_TOOLCHAIN
    ADDITIONAL_HEADERS ${public_headers} ${private_headers}
  )

  enable_cuda_compilation(flang_rt.static "${supported_sources}")
  enable_omp_offload_compilation(flang_rt.static "${supported_sources}")

  # Select a default runtime, which is used for unittests.
  if (TARGET flang_rt.shared AND BUILD_SHARED_LIBS)
    set(default_flang_rt "flang_rt.shared")
  elseif (TARGET flang_rt.static)
    set(default_flang_rt "flang_rt.static")
  else ()
    set(default_flang_rt "flang_rt.shared")
  endif ()
  add_library(flang_rt ALIAS "${default_flang_rt}")
  add_library(flang_rt.unittest ALIAS "${default_flang_rt}")
else()
  # Target for building all versions of the runtime
  add_custom_target(flang_rt)
  set_target_properties(flang_rt PROPERTIES FOLDER "Flang-RT/Meta")

  function (add_win_flangrt_library libtype suffix msvc_lib)
    set(name "flang_rt.${suffix}")
    add_flangrt_library(${name} ${libtype}
        ${sources}
        ${ARGN}
        ADDITIONAL_HEADERS ${public_headers} ${private_headers}
      )

    if (msvc_lib)
      set_target_properties(${name}
          PROPERTIES
            MSVC_RUNTIME_LIBRARY "${msvc_lib}"
        )
    endif ()

    # Setting an unique Fortran_MODULE_DIRECTORY is required for each variant to
    # write a different .mod file.
    set_target_properties(${name}
        PROPERTIES
          Fortran_MODULE_DIRECTORY "module.${suffix}"
      )

    enable_cuda_compilation(${name} "${supported_sources}")
    enable_omp_offload_compilation(${name} "${supported_sources}")
    add_dependencies(flang_rt ${name})
  endfunction ()

  # Variants of the static flang_rt for different versions of the msvc runtime.
  #
  # The dynamic/dynamic_dbg variants are not DLLs themselves, only require
  # linking to msvcrt(d).dll.
  # FIXME: Generating actual runtime DLLs is currently not possible. There are
  # two roadblocks:
  #
  #  * Flang emits /DEFAULTLIB:flang_rt.dynamic.lib into
  #    iso_fortran_env_impl.f90.obj. Because that file is itself part of
  #    flang_rt.dynamic, this results in a recursive dependency when invoking
  #    the linker.
  #
  #  * The externally-visible functions must either be annotated with
  #    __declspec(dllexport), or listed in an exports file. A possible workaround
  #    is CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS which would also export the internal
  #    C++ symbols and still requires global data symbols to be annotated
  #    manually.
  add_win_flangrt_library(STATIC static      MultiThreaded         INSTALL_WITH_TOOLCHAIN)
  add_win_flangrt_library(STATIC static_dbg  MultiThreadedDebug    INSTALL_WITH_TOOLCHAIN)
  add_win_flangrt_library(STATIC dynamic     MultiThreadedDLL      INSTALL_WITH_TOOLCHAIN)
  add_win_flangrt_library(STATIC dynamic_dbg MultiThreadedDebugDLL INSTALL_WITH_TOOLCHAIN)

  # Unittests link against LLVMSupport which is using CMake's default runtime
  # library selection, which is either MultiThreadedDLL or MultiThreadedDebugDLL
  # depending on the configuration. They have to match or linking will fail.
  if (GENERATOR_IS_MULTI_CONFIG)
    # We cannot select an ALIAS library because it may be different
    # per configuration. Fallback to CMake's default.
    add_win_flangrt_library(STATIC unittest "" EXCLUDE_FROM_ALL)
  else ()
    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if (build_type STREQUAL "debug")
      add_library(flang_rt.unittest ALIAS flang_rt.dynamic_dbg)
    else ()
      add_library(flang_rt.unittest ALIAS flang_rt.dynamic)
    endif ()
  endif ()
endif()
