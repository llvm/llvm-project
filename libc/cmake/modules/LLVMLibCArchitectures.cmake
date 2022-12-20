# ------------------------------------------------------------------------------
# Architecture and OS definitions.
#
# The correct target OS and architecture to build the libc for is deduced here.
# When possible, we also setup appropriate compile options for the target
# platform.
# ------------------------------------------------------------------------------

if(LIBC_GPU_BUILD)
  # We set the generic target and OS to "gpu" here. More specific defintions
  # for the exact target GPU are set up in prepare_libc_gpu_build.cmake.
  set(LIBC_TARGET_OS "gpu")
  set(LIBC_TARGET_ARCHITECTURE_IS_GPU TRUE)
  set(LIBC_TARGET_ARCHITECTURE "gpu")
  if(LIBC_TARGET_TRIPLE)
    message(WARNING "LIBC_TARGET_TRIPLE is ignored as LIBC_GPU_BUILD is on. ")
  endif()
  return()
endif()

if(MSVC)
  # If the compiler is visual c++ or equivalent, we will assume a host build.
  set(LIBC_TARGET_OS ${CMAKE_HOST_SYSTEM_NAME})
  string(TOLOWER ${LIBC_TARGET_OS} LIBC_TARGET_OS)
  set(LIBC_TARGET_ARCHITECTURE ${CMAKE_HOST_SYSTEM_PROCESSOR})
  if(LIBC_TARGET_TRIPLE)
    message(WARNING "libc build: Detected MSVC or equivalent compiler; "
                    "LIBC_TARGET_TRIPLE is ignored and a host build is assumed.")
  endif()
  return()
endif()

# A helper function to get the architecture and system components from a target
# triple.
function(get_arch_and_system_from_triple triple arch_var sys_var)
  string(REPLACE "-" ";" triple_comps ${triple})
  list(LENGTH triple_comps triple_size)
  if(triple_size LESS "3")
    return()
  endif()
  math(EXPR system_index "${triple_size} - 2")
  list(GET triple_comps 0 target_arch)
  # The target_arch string can have sub-architecture suffixes which we want to
  # remove. So, we regex-match the string and set target_arch to a cleaner
  # value.
  if(target_arch MATCHES "^mips")
    set(target_arch "mips")
  elseif(target_arch MATCHES "^arm")
    set(target_arch "arm")
  elseif(target_arch MATCHES "^aarch64")
    set(target_arch "aarch64")
  elseif(target_arch MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
    set(target_arch "x86_64")
  elseif(target_arch MATCHES "^(powerpc|ppc)")
    set(target_arch "power")
  else()
    return()
  endif()

  set(${arch_var} ${target_arch} PARENT_SCOPE)
  list(GET triple_comps ${system_index} target_sys)
  set(${sys_var} ${target_sys} PARENT_SCOPE)
endfunction(get_arch_and_system_from_triple)

# Query the default target triple of the compiler.
set(target_triple_option "-print-target-triple")
if(CMAKE_COMPILER_IS_GNUCXX)
  # GCC does not support the "-print-target-triple" option but supports
  # "-print-multiarch" which clang does not support for all targets.
  set(target_triple_option "-print-multiarch")
endif()
execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${target_triple_option}
                RESULT_VARIABLE libc_compiler_triple_check
                OUTPUT_VARIABLE libc_compiler_triple)
if(NOT (libc_compiler_triple_check EQUAL "0"))
  message(FATAL_ERROR "libc build: error querying target triple from the "
                      "compiler: ${libc_compiler_triple}")
endif()
get_arch_and_system_from_triple(${libc_compiler_triple}
                                compiler_arch compiler_sys)
if(NOT compiler_arch)
  message(FATAL_ERROR
          "libc build: Invalid or unknown libc compiler target triple: "
          "${libc_compiler_triple}")
endif()

set(LIBC_TARGET_ARCHITECTURE ${compiler_arch})
set(LIBC_TARGET_OS ${compiler_sys})

# The libc's target architecture and OS are set to match the compiler's default
# target triple above. However, one can explicitly set LIBC_TARGET_TRIPLE. If it
# is and does not match the compiler's target triple, then we will use it set up
# libc's target architecture and OS.
if(LIBC_TARGET_TRIPLE)
  get_arch_and_system_from_triple(${LIBC_TARGET_TRIPLE} libc_arch libc_sys)
  if(NOT libc_arch)
    message(FATAL_ERROR
            "libc build: Invalid or unknown triple in LIBC_TARGET_TRIPLE: "
            "${LIBC_TARGET_TRIPLE}")
  endif()
  set(LIBC_TARGET_ARCHITECTURE ${libc_arch})
  set(LIBC_TARGET_OS ${libc_sys})
endif()

if((LIBC_TARGET_OS STREQUAL "unknown") OR (LIBC_TARGET_OS STREQUAL "none"))
  # We treat "unknown" and "none" systems as baremetal targets.
  set(LIBC_TARGET_OS "baremetal")
endif()

# Set up some convenient vars to make conditionals easy to use in other parts of
# the libc CMake infrastructure. Also, this is where we also check if the target
# architecture is currently supported.
if(LIBC_TARGET_ARCHITECTURE STREQUAL "arm")
  set(LIBC_LIBC_TARGET_ARCHITECTUREITECTURE_IS_ARM TRUE)
elseif(LIBC_TARGET_ARCHITECTURE STREQUAL "aarch64")
  set(LIBC_LIBC_TARGET_ARCHITECTUREITECTURE_IS_AARCH64 TRUE)
elseif(LIBC_TARGET_ARCHITECTURE STREQUAL "x86_64")
  set(LIBC_LIBC_TARGET_ARCHITECTUREITECTURE_IS_X86 TRUE)
else()
  message(FATAL_ERROR
          "Unsupported libc target architecture ${LIBC_TARGET_ARCHITECTURE}")
endif()

# If the compiler target triple is not the same as the triple specified by
# LIBC_TARGET_TRIPLE, we will add a --target option if the compiler is clang.
# If the compiler is GCC we just error out as there is no equivalent of an
# option like --target.
if(LIBC_TARGET_TRIPLE AND
   (NOT (libc_compiler_triple STREQUAL LIBC_TARGET_TRIPLE)))
  if(CMAKE_COMPILER_IS_GNUCXX)
    message(FATAL_ERROR
            "GCC target triple and LIBC_TARGET_TRIPLE do not match.")
  else()
    list(APPEND LIBC_COMPILE_OPTIONS_DEFAULT "--target=${LIBC_TARGET_TRIPLE}")
  endif()
endif()

message(STATUS
        "Building libc for ${LIBC_TARGET_ARCHITECTURE} on ${LIBC_TARGET_OS}")
