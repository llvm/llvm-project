if(NOT CMAKE_CLC_COMPILER)
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "The CLC language requires the C compiler (CMAKE_C_COMPILER) to be "
      "Clang, but CMAKE_C_COMPILER_ID is '${CMAKE_C_COMPILER_ID}'.")
  endif()

  # Use the regular clang driver if the C compiler is clang-cl.
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang" AND CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    cmake_path(GET CMAKE_C_COMPILER PARENT_PATH llvm_bin_dir)
    find_program(clang_exe clang
      HINTS "${llvm_bin_dir}"
      NO_DEFAULT_PATH
    )
    if(NOT clang_exe)
      message(FATAL_ERROR "clang-cl detected, but clang not found in ${llvm_bin_dir}")
    endif()
    set(clc_compiler "${clang_exe}")
  else()
    set(clc_compiler "${CMAKE_C_COMPILER}")
  endif()
  set(CMAKE_CLC_COMPILER "${clc_compiler}" CACHE FILEPATH "libclc: CLC compiler")
endif()

mark_as_advanced(CMAKE_CLC_COMPILER)

set(CMAKE_CLC_COMPILER_ID "Clang")
set(CMAKE_CLC_COMPILER_ID_RUN TRUE)

configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeCLCCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeCLCCompiler.cmake @ONLY)

set(CMAKE_CLC_COMPILER_ENV_VAR "CLC")
