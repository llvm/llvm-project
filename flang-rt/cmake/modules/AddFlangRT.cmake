#===-- cmake/modules/AddFlangRT.cmake --------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# Builds a library with common options for Flang-RT.
#
# Usage:
#
# add_flangrt_library(name sources ...
#   SHARED
#     Build a dynamic (.so/.dll) library
#   STATIC
#     Build a static (.a/.lib) library
#   OBJECT
#     Always create an object library.
#     Without SHARED/STATIC, build only the object library.
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
#   INCLUDE_DIRECTORIES
#     Additional target_include_directories for all added targets
#   LINK_LIBRARIES
#     Additional target_link_libraries for all added targets
#   TARGET_PROPERTIES
#     Set target properties of all added targets
# )
function (add_flangrt_library name)
  set(options STATIC SHARED OBJECT INSTALL_WITH_TOOLCHAIN EXCLUDE_FROM_ALL LINK_TO_LLVM)
  set(multiValueArgs ADDITIONAL_HEADERS INCLUDE_DIRECTORIES LINK_LIBRARIES TARGET_PROPERTIES)
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

  # Internal names of libraries. If called with just single type option, use
  # the default name for it. Name of targets must only depend on function
  # arguments to be predictable for callers.
  set(name_static "${name}.static")
  set(name_shared "${name}.shared")
  set(name_object "obj.${name}")
  if (ARG_STATIC AND NOT ARG_SHARED)
    set(name_static "${name}")
  elseif (NOT ARG_STATIC AND ARG_SHARED)
    set(name_shared "${name}")
  elseif (NOT ARG_STATIC AND NOT ARG_SHARED AND ARG_OBJECT)
    set(name_object "${name}")
  elseif (NOT ARG_STATIC AND NOT ARG_SHARED AND NOT ARG_OBJECT)
    # Only one of them will actually be built.
    set(name_static "${name}")
    set(name_shared "${name}")
  endif ()

  # Determine what to build. If not explicitly specified, honor
  # BUILD_SHARED_LIBS (e.g. for unittest libraries). If can build static and
  # shared, use ENABLE_STATIC/ENABLE_SHARED setting.
  if (ARG_STATIC AND ARG_SHARED)
    set(build_static ${FLANG_RT_ENABLE_STATIC})
    set(build_shared ${FLANG_RT_ENABLE_SHARED})
  else ()
    set(build_static ${ARG_STATIC})
    set(build_shared ${ARG_SHARED})
  endif ()
  if (NOT ARG_STATIC AND NOT ARG_SHARED AND NOT ARG_OBJECT)
    if (BUILD_SHARED_LIBS)
      set(build_shared ON)
    else ()
      set(build_static ON)
    endif ()
  endif ()

  # Build an object library if building multiple libraries at once or if
  # explicitly requested.
  set(build_object OFF)
  if (ARG_OBJECT)
    set(build_object ON)
  elseif (build_static AND build_shared)
    set(build_object ON)
  endif ()

  # srctargets: targets that contain source files
  # libtargets: static/shared if they are built
  # alltargets: any add_library target added by this function
  set(srctargets "")
  set(libtargets "")
  set(alltargets "")
  if (build_static)
    list(APPEND srctargets "${name_static}")
    list(APPEND libtargets "${name_static}")
    list(APPEND alltargets "${name_static}")
  endif ()
  if (build_shared)
    list(APPEND srctargets "${name_shared}")
    list(APPEND libtargets "${name_shared}")
    list(APPEND alltargets "${name_shared}")
  endif ()
  if (build_object)
    set(srctargets "${name_object}")
    list(APPEND alltargets "${name_object}")
  endif ()

  set(extra_args "")
  if (ARG_EXCLUDE_FROM_ALL)
    list(APPEND extra_args EXCLUDE_FROM_ALL)
  endif ()

  # Also add header files to IDEs to list as part of the library.
  set_source_files_properties(${ARG_ADDITIONAL_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)

  # Create selected library types.
  if (build_object)
    add_library("${name_object}" OBJECT ${extra_args} ${ARG_ADDITIONAL_HEADERS} ${ARG_UNPARSED_ARGUMENTS})
    set_target_properties(${name_object} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        FOLDER "Flang-RT/Object Libraries"
      )

    # Replace arguments for the libraries we are going to create.
    set(ARG_ADDITIONAL_HEADERS "")
    set(ARG_UNPARSED_ARGUMENTS "$<TARGET_OBJECTS:${name_object}>")
  endif ()
  if (build_static)
    add_library("${name_static}" STATIC ${extra_args} ${ARG_ADDITIONAL_HEADERS} ${ARG_UNPARSED_ARGUMENTS})
    target_link_libraries("${name_static}" PRIVATE flang-rt-libcxx-headers flang-rt-libc-headers flang-rt-libc-static)
  endif ()
  if (build_shared)
    add_library("${name_shared}" SHARED ${extra_args} ${ARG_ADDITIONAL_HEADERS} ${ARG_UNPARSED_ARGUMENTS})
    target_link_libraries("${name_shared}" PRIVATE flang-rt-libcxx-headers flang-rt-libc-headers flang-rt-libc-shared)
    if (Threads_FOUND) 
      target_link_libraries(${name_shared} PUBLIC Threads::Threads)
    endif ()

    # Special dependencies handling for shared libraries only:
    #
    # flang-rt libraries must not depend on libc++/libstdc++,
    # so set the linker language to C to avoid the unnecessary
    # library dependence. Note that libc++/libstdc++ may still
    # come through CMAKE_CXX_IMPLICIT_LINK_LIBRARIES.
    set_target_properties(${name_shared} PROPERTIES LINKER_LANGUAGE C)
    # Use --as-needed to avoid unnecessary dependencies.
    if (LINKER_AS_NEEDED_OPT)
      target_link_options(${name_shared} BEFORE PRIVATE
          "${LINKER_AS_NEEDED_OPT}"
        )
    endif()
  endif ()

  if (libtargets)
    # Provide a default alias which exists in either setting.
    if (BUILD_SHARED_LIBS)
      if (build_shared)
        set(default_target "${name_shared}")
      else ()
        set(default_target "${name_static}")
      endif ()
    else ()
      if (build_static)
        set(default_target "${name_static}")
      else ()
        set(default_target "${name_shared}")
      endif ()
    endif ()
    add_library(${name}.default ALIAS "${default_target}")

    # Provide a build target that builds any enabled library.
    # Not intended for target_link_libraries. Either use the ${name}.static,
    # ${name}.shared variants, or ${name}.default to let BUILD_SHARED_LIBS
    # decide.
    if (NOT TARGET ${name})
      add_custom_target(${name})
      add_dependencies(${name} ${libtargets})
    endif ()
  endif ()

  foreach (tgtname IN LISTS libtargets)
    if (NOT WIN32)
      # Use same stem name for .a and .so. Common in UNIX environments.
      # Not possible in Windows environments.
      set_target_properties(${tgtname} PROPERTIES OUTPUT_NAME "${name}")
    endif ()

    if (ARG_INSTALL_WITH_TOOLCHAIN)
      set_target_properties(${tgtname} PROPERTIES FOLDER "Flang-RT/Toolchain Libraries")
    else ()
      set_target_properties(${tgtname} PROPERTIES FOLDER "Flang-RT/Libraries")
    endif ()
  endforeach ()

 set(TARGET_FLAGS)
  if(APPLE)
    set(DARWIN_EMBEDDED_PLATFORMS)
    set(DARWIN_osx_BUILTIN_MIN_VER 10.7)
    set(DARWIN_osx_BUILTIN_MIN_VER_FLAG
        -mmacosx-version-min=${DARWIN_osx_BUILTIN_MIN_VER})
  endif()

  # Define how to compile and link the library.
  # Some conceptionally only apply to ${srctargets} or ${libtargets}, but we
  # apply them to ${alltargets}. In worst case, they are ignored by CMake.
  foreach (tgtname IN LISTS alltargets)
    # Minimum required C++ version for Flang-RT, even if CMAKE_CXX_STANDARD is defined to something else.
    target_compile_features(${tgtname} PRIVATE cxx_std_17)

    # When building the flang runtime if LTO is enabled the archive file
    # contains LLVM IR rather than object code. Currently flang is not
    # LTO aware so cannot link this file to compiled Fortran code.
    if (FLANG_RT_HAS_FNO_LTO_FLAG)
      target_compile_options(${tgtname} PRIVATE -fno-lto)
    endif ()

    # Use compiler-specific options to disable exceptions and RTTI.
    if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
      target_compile_options(${tgtname} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti -funwind-tables -fno-asynchronous-unwind-tables>
        )
    elseif (MSVC)
      target_compile_options(${tgtname} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:/EHs-c- /GR->
        )
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "XL")
      target_compile_options(${tgtname} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-qnoeh -qnortti>
        )
    endif ()

    # Add target specific options if necessary.
    if ("${LLVM_RUNTIMES_TARGET}" MATCHES "^amdgcn")
      target_compile_options(${tgtname} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-nogpulib -flto -fvisibility=hidden>
        )
    elseif ("${LLVM_RUNTIMES_TARGET}" MATCHES "^nvptx")
      target_compile_options(${tgtname} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-nogpulib -flto -fvisibility=hidden -Wno-unknown-cuda-version --cuda-feature=+ptx63>
        )
    elseif (APPLE)
      target_compile_options(${tgtname} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:${DARWIN_osx_BUILTIN_MIN_VER_FLAG}>
        )
    endif ()

    # Also for CUDA source when compiling with FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT=CUDA
    if (CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
      # Assuming gcc as host compiler.
      target_compile_options(${tgtname} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions -Xcompiler -fno-rtti -Xcompiler -fno-unwind-tables -Xcompiler -fno-asynchronous-unwind-tables>
        )
    else ()
      # Assuming a clang-compatible CUDA compiler.
      target_compile_options(${tgtname} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-fno-exceptions -fno-rtti -fno-unwind-tables -fno-asynchronous-unwind-tables>
        )
    endif ()

    # Flang-RT's public headers
    target_include_directories(${tgtname} PUBLIC "${FLANG_RT_SOURCE_DIR}/include")

    # For ISO_Fortran_binding.h to be found by the runtime itself (Accessed as #include "flang/ISO_Fortran_binding.h")
    # User applications can use #include <ISO_Fortran_binding.h>
    target_include_directories(${tgtname} PUBLIC "${FLANG_SOURCE_DIR}/include")

    # For Flang-RT's configured config.h to be found
    target_include_directories(${tgtname} PRIVATE "${FLANG_RT_BINARY_DIR}")

    # Disable libstdc++/libc++ assertions, even in an LLVM_ENABLE_ASSERTIONS
    # build, to avoid an unwanted dependency on libstdc++/libc++.so.
    target_compile_definitions(${tgtname} PUBLIC _GLIBCXX_NO_ASSERTIONS)
    if (FLANG_RT_SUPPORTS_UNDEFINE_FLAG)
      target_compile_options(${tgtname} PUBLIC -U_GLIBCXX_ASSERTIONS)
      target_compile_options(${tgtname} PUBLIC -U_LIBCPP_ENABLE_ASSERTIONS)
    endif ()

    # Non-GTest unittests depend on LLVMSupport
    if (ARG_LINK_TO_LLVM)
      if (LLVM_LINK_LLVM_DYLIB)
        set(llvm_libs LLVM)
      else()
        llvm_map_components_to_libnames(llvm_libs Support)
      endif()
      target_link_libraries(${tgtname} PUBLIC ${llvm_libs})
      target_include_directories(${tgtname} PUBLIC ${LLVM_INCLUDE_DIRS})
    endif ()

    if (ARG_INCLUDE_DIRECTORIES)
      target_include_directories(${tgtname} ${ARG_INCLUDE_DIRECTORIES})
    endif ()

    if (ARG_LINK_LIBRARIES)
      target_link_libraries(${tgtname} PUBLIC ${ARG_LINK_LIBRARIES})
    endif ()
  endforeach ()

  foreach (tgtname IN LISTS libtargets)
    # If this is part of the toolchain, put it into the compiler's resource
    # directory. Otherwise it is part of testing and is not installed at all.
    # TODO: Consider multi-configuration builds (MSVC_IDE, "Ninja Multi-Config")
    if (ARG_INSTALL_WITH_TOOLCHAIN)
      set_target_properties(${tgtname}
        PROPERTIES
          ARCHIVE_OUTPUT_DIRECTORY "${FLANG_RT_OUTPUT_RESOURCE_LIB_DIR}"
          LIBRARY_OUTPUT_DIRECTORY "${FLANG_RT_OUTPUT_RESOURCE_LIB_DIR}"
        )

      install(TARGETS ${tgtname}
          ARCHIVE DESTINATION "${FLANG_RT_INSTALL_RESOURCE_LIB_PATH}"
          LIBRARY DESTINATION "${FLANG_RT_INSTALL_RESOURCE_LIB_PATH}"
        )
    endif ()

    if (ARG_TARGET_PROPERTIES)
      set_target_properties(${tgtname} PROPERTIES ${ARG_TARGET_PROPERTIES})
    endif ()

    # flang-rt should build all the Flang-RT targets that are built in an
    # 'all' build.
    if (NOT ARG_EXCLUDE_FROM_ALL)
      add_dependencies(flang-rt ${tgtname})
    endif ()
  endforeach ()
endfunction (add_flangrt_library)
