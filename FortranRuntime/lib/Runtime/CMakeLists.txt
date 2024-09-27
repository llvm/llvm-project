#===-- lib/Runtime/CMakeLists.txt ------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_subdirectory(Float128Math)

set(sources
  ISO_Fortran_binding.cpp
  allocator-registry.cpp
  allocatable.cpp
  array-constructor.cpp
  assign.cpp
  buffer.cpp
  character.cpp
  command.cpp
  complex-powi.cpp
  complex-reduction.c
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
  exceptions.cpp
  execute.cpp
  extensions.cpp
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
  iso_fortran_env_impl.f90
  iostat.cpp
  main.cpp
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
  random.cpp
  reduce.cpp
  reduction.cpp
  stat.cpp
  stop.cpp
  sum.cpp
  support.cpp
  temporary-stack.cpp
  terminator.cpp
  time-intrinsic.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  type-info.cpp
  unit-map.cpp
  unit.cpp
  utf.cpp
  ../Common/binary-to-decimal.cpp
  ../Common/decimal-to-binary.cpp
)

# List of files that are buildable for all devices.
set(supported_files
  ISO_Fortran_binding.cpp
  allocatable.cpp
  allocator-registry.cpp
  array-constructor.cpp
  assign.cpp
  buffer.cpp
  character.cpp
  connection.cpp
  copy.cpp
  derived-api.cpp
  derived.cpp
  descriptor.cpp
  descriptor-io.cpp
  dot-product.cpp
  edit-input.cpp
  edit-output.cpp
  environment.cpp
  extrema.cpp
  external-unit.cpp
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
  utf.cpp
  )


  set(public_headers "")
  file(GLOB_RECURSE public_headers 
    "${FLANGRUNTIME_SOURCE_DIR}/include/flang/Runtime/*.h" 
    "${FLANGRUNTIME_SOURCE_DIR}/include/flang/Common/*.h" 
    )

  set(private_headers "")
  file(GLOB_RECURSE private_headers 
    "${FLANGRUNTIME_SOURCE_DIR}/lib/Runtime/*.h" 
    "${FLANGRUNTIME_SOURCE_DIR}/lib/Common/*.h" 
    )

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
  list(APPEND sources ${f128_sources})
endif()

if (NOT DEFINED WIN32)
  add_fortranruntime_library(FortranRuntime STATIC
    ${sources}
    INSTALL_WITH_TOOLCHAIN
    ADDITIONAL_HEADERS ${public_headers} ${private_headers}
  )

  # For unittests that link to FortranRuntime. Should link to the static version of the library.
  add_library(FortranRuntime.static ALIAS FortranRuntime)
else()
  # ${sources} also contains Fortran files. CMake write the generated Fortran files into the same directory (lib/Runtime/iso_fortran_env_impl.mod), so we need to compile it only once and add the object files to different runtime version.
  add_fortranruntime_library(FortranRuntime.obj OBJECT
      ${sources}
 
      ADDITIONAL_HEADERS ${public_headers} ${private_headers}
    )

  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
  add_fortranruntime_library(FortranRuntime.static STATIC $<TARGET_OBJECTS:FortranRuntime.obj> INSTALL_WITH_TOOLCHAIN  #${sources}
    )

  
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  add_fortranruntime_library(FortranRuntime.dynamic SHARED $<TARGET_OBJECTS:FortranRuntime.obj> INSTALL_WITH_TOOLCHAIN #${sources} 
    )

  
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebug)
  add_fortranruntime_library(FortranRuntime.static_dbg STATIC $<TARGET_OBJECTS:FortranRuntime.obj> INSTALL_WITH_TOOLCHAIN #${sources}
    )

  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
  add_fortranruntime_library(FortranRuntime.dynamic_dbg SHARED $<TARGET_OBJECTS:FortranRuntime.obj> INSTALL_WITH_TOOLCHAIN #${sources}
    )
  target_link_libraries(FortranRuntime.dynamic_dbg PRIVATE FortranRuntime.obj)

  # Target for building all versions of the runtime
  add_custom_target(FortranRuntime)
  set_target_properties(FortranRuntime PROPERTIES FOLDER "Fortran Runtime/Meta")
  add_dependencies(FortranRuntime FortranRuntime.static FortranRuntime.dynamic
    FortranRuntime.static_dbg FortranRuntime.dynamic_dbg)

  if (MSVC)
    # The Fortran files compiled with flang-new assumes libcall functions such as `__udivti3`
    # Add LLVM_ENABLE_RUNTIMES=compiler-rt for it to be available during the runtimes build
    # FIXME: Can embed linker command --dependent-lib into object files as already done with FortranRuntime.<mode>.lib
    if (NOT TARGET clang_rt.builtins-x86_64)
      message(WARNING
        "compiler-rt needed when compiling with msvc\n"
        "Flang/LLVM will emit code that calls into libgcc/compiler-rt, but using the MSVC linker will only link Microsoft's CRT automatically\n"
        "Add compiler-rt to LLVM_ENABLE_RUNTIMES to fix"
      )
    endif ()
    target_link_libraries(FortranRuntime.static PUBLIC clang_rt.builtins-x86_64)
    target_link_libraries(FortranRuntime.dynamic PUBLIC clang_rt.builtins-x86_64)
    target_link_libraries(FortranRuntime.static_dbg PUBLIC clang_rt.builtins-x86_64)
    target_link_libraries(FortranRuntime.dynamic_dbg PUBLIC clang_rt.builtins-x86_64)
  endif ()
endif()
