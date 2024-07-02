# HandleLibcxxFlags - A set of macros used to setup the flags used to compile
# and link libc++abi. These macros add flags to the following CMake variables.
# - LIBCXXABI_COMPILE_FLAGS: flags used to compile libc++abi
# - LIBCXXABI_LINK_FLAGS: flags used to link libc++abi
# - LIBCXXABI_LIBRARIES: libraries to link libc++abi to.

include(CheckCXXCompilerFlag)
include(HandleFlags)

unset(add_flag_if_supported)

# Add a specified list of flags to both 'LIBCXXABI_COMPILE_FLAGS' and
# 'LIBCXXABI_LINK_FLAGS'.
macro(add_flags)
  foreach(value ${ARGN})
    list(APPEND LIBCXXABI_COMPILE_FLAGS ${value})
    list(APPEND LIBCXXABI_LINK_FLAGS ${value})
  endforeach()
endmacro()

# If the specified 'condition' is true then add a list of flags to both
# 'LIBCXXABI_COMPILE_FLAGS' and 'LIBCXXABI_LINK_FLAGS'.
macro(add_flags_if condition)
  if (${condition})
    add_flags(${ARGN})
  endif()
endmacro()

# Add each flag in the list to LIBCXXABI_COMPILE_FLAGS and LIBCXXABI_LINK_FLAGS
# if that flag is supported by the current compiler.
macro(add_flags_if_supported)

  # Disable linker for CMake flag compatibility checks
  #
  # Due to https://gitlab.kitware.com/cmake/cmake/-/issues/23454, we need to
  # disable CMAKE_REQUIRED_LINK_OPTIONS (c.f. CXX_SUPPORTS_UNWINDLIB_EQ_NONE_FLAG),
  # for static targets; cache the target type here, and reset it after the various
  # checks have been performed.
  set(_previous_CMAKE_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
  set(_previous_CMAKE_REQUIRED_LINK_OPTIONS ${CMAKE_REQUIRED_LINK_OPTIONS})
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(CMAKE_REQUIRED_LINK_OPTIONS)

  foreach(flag ${ARGN})
      mangle_name("${flag}" flagname)
      check_cxx_compiler_flag("${flag}" "CXX_SUPPORTS_${flagname}_FLAG")
      add_flags_if(CXX_SUPPORTS_${flagname}_FLAG ${flag})
  endforeach()

  # reset CMAKE_TRY_COMPILE_TARGET_TYPE & CMAKE_REQUIRED_LINK_OPTIONS after flag checks
  set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_previous_CMAKE_TRY_COMPILE_TARGET_TYPE})
  set(CMAKE_REQUIRED_LINK_OPTIONS ${_previous_CMAKE_REQUIRED_LINK_OPTIONS})
endmacro()

# Add a list of flags to 'LIBCXXABI_COMPILE_FLAGS'.
macro(add_compile_flags)
  foreach(f ${ARGN})
    list(APPEND LIBCXXABI_COMPILE_FLAGS ${f})
  endforeach()
endmacro()

# If 'condition' is true then add the specified list of flags to
# 'LIBCXXABI_COMPILE_FLAGS'
macro(add_compile_flags_if condition)
  if (${condition})
    add_compile_flags(${ARGN})
  endif()
endmacro()

# For each specified flag, add that flag to 'LIBCXXABI_COMPILE_FLAGS' if the
# flag is supported by the C++ compiler.
macro(add_compile_flags_if_supported)
  foreach(flag ${ARGN})
      mangle_name("${flag}" flagname)
      check_cxx_compiler_flag("${flag}" "CXX_SUPPORTS_${flagname}_FLAG")
      add_compile_flags_if(CXX_SUPPORTS_${flagname}_FLAG ${flag})
  endforeach()
endmacro()

# For each specified flag, add that flag to 'LIBCXXABI_COMPILE_FLAGS' if the
# flag is supported by the C compiler.
macro(add_c_compile_flags_if_supported)
  foreach(flag ${ARGN})
      mangle_name("${flag}" flagname)
      check_c_compiler_flag("${flag}" "C_SUPPORTS_${flagname}_FLAG")
      add_compile_flags_if(C_SUPPORTS_${flagname}_FLAG ${flag})
  endforeach()
endmacro()

# Add a list of flags to 'LIBCXXABI_LINK_FLAGS'.
macro(add_link_flags)
  foreach(f ${ARGN})
    list(APPEND LIBCXXABI_LINK_FLAGS ${f})
  endforeach()
endmacro()

# If 'condition' is true then add the specified list of flags to
# 'LIBCXXABI_LINK_FLAGS'
macro(add_link_flags_if condition)
  if (${condition})
    add_link_flags(${ARGN})
  endif()
endmacro()

# For each specified flag, add that flag to 'LIBCXXABI_LINK_FLAGS' if the
# flag is supported by the C++ compiler.
macro(add_link_flags_if_supported)
  foreach(flag ${ARGN})
    mangle_name("${flag}" flagname)
    check_cxx_compiler_flag("${flag}" "CXX_SUPPORTS_${flagname}_FLAG")
    add_link_flags_if(CXX_SUPPORTS_${flagname}_FLAG ${flag})
  endforeach()
endmacro()

# Add a list of libraries or link flags to 'LIBCXXABI_LIBRARIES'.
macro(add_library_flags)
  foreach(lib ${ARGN})
    list(APPEND LIBCXXABI_LIBRARIES ${lib})
  endforeach()
endmacro()

# if 'condition' is true then add the specified list of libraries and flags
# to 'LIBCXXABI_LIBRARIES'.
macro(add_library_flags_if condition)
  if(${condition})
    add_library_flags(${ARGN})
  endif()
endmacro()
