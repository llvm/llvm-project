# ------------------------------------------------------------------------------
# Architecture and OS definitions.
#
# The correct target OS and architecture to build the libc for is deduced here.
# When possible, we also setup appropriate compile options for the target
# platform.
# ------------------------------------------------------------------------------

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
    # TODO(lntue): Shall we separate `arm64`?  It is currently recognized as
    # `arm` here.
    set(target_arch "arm")
  elseif(target_arch MATCHES "^aarch64")
    set(target_arch "aarch64")
  elseif(target_arch MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
    set(target_arch "x86_64")
  elseif(target_arch MATCHES "^(powerpc|ppc)")
    set(target_arch "power")
  elseif(target_arch MATCHES "^riscv32")
    set(target_arch "riscv32")
  elseif(target_arch MATCHES "^riscv64")
    set(target_arch "riscv64")
  elseif(target_arch MATCHES "^amdgcn")
    set(target_arch "amdgpu")
  elseif(target_arch MATCHES "^nvptx64")
    set(target_arch "nvptx")
  else()
    return()
  endif()

  set(${arch_var} ${target_arch} PARENT_SCOPE)
  list(GET triple_comps ${system_index} target_sys)

  # Correcting OS name for Apple's systems.
  if(target_sys STREQUAL "apple")
    list(GET triple_comps 2 target_sys)
  endif()
  # Strip version from `darwin###`
  if(target_sys MATCHES "^darwin")
    set(target_sys "darwin")
  endif()

  # Setting OS name for GPU architectures.
  list(GET triple_comps -1 gpu_target_sys)
  if(gpu_target_sys MATCHES "^amdhsa" OR gpu_target_sys MATCHES "^cuda")
    set(target_sys "gpu")
  endif()

  set(${sys_var} ${target_sys} PARENT_SCOPE)
endfunction(get_arch_and_system_from_triple)

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version -v
                RESULT_VARIABLE libc_compiler_info_result
                OUTPUT_VARIABLE libc_compiler_info
                ERROR_VARIABLE libc_compiler_info)
if(NOT (libc_compiler_info_result EQUAL "0"))
  message(FATAL_ERROR "libc build: error querying compiler info from the "
                      "compiler: ${libc_compiler_info}")
endif()
string(REGEX MATCH "Target: [-_a-z0-9.]+[ \r\n]+"
       libc_compiler_target_info ${libc_compiler_info})
if(NOT libc_compiler_target_info)
  message(FATAL_ERROR "libc build: could not read compiler target info from:\n"
                      "${libc_compiler_info}")
endif()
string(STRIP ${libc_compiler_target_info} libc_compiler_target_info)
string(SUBSTRING ${libc_compiler_target_info} 8 -1 libc_compiler_triple)
get_arch_and_system_from_triple(${libc_compiler_triple}
                                compiler_arch compiler_sys)
if(NOT compiler_arch)
  message(FATAL_ERROR
          "libc build: Invalid or unknown libc compiler target triple: "
          "${libc_compiler_triple}")
endif()

set(LIBC_TARGET_ARCHITECTURE ${compiler_arch})
set(LIBC_TARGET_OS ${compiler_sys})
set(LIBC_CROSSBUILD FALSE)

# One should not set LLVM_RUNTIMES_TARGET and LIBC_TARGET_TRIPLE
if(LLVM_RUNTIMES_TARGET AND LIBC_TARGET_TRIPLE)
  message(FATAL_ERROR
          "libc build: Specify only LLVM_RUNTIMES_TARGET if you are doing a "
          "runtimes/bootstrap build. If you are doing a standalone build, "
          "specify only LIBC_TARGET_TRIPLE.")
endif()

set(explicit_target_triple)
if(LLVM_RUNTIMES_TARGET)
  set(explicit_target_triple ${LLVM_RUNTIMES_TARGET})
elseif(LIBC_TARGET_TRIPLE)
  set(explicit_target_triple ${LIBC_TARGET_TRIPLE})
endif()

# The libc's target architecture and OS are set to match the compiler's default
# target triple above. However, one can explicitly set LIBC_TARGET_TRIPLE or
# LLVM_RUNTIMES_TARGET (for runtimes/bootstrap build). If one of them is set,
# then we will use that target triple to deduce libc's target OS and
# architecture.
if(explicit_target_triple)
  get_arch_and_system_from_triple(${explicit_target_triple} libc_arch libc_sys)
  if(NOT libc_arch)
    message(FATAL_ERROR
            "libc build: Invalid or unknown triple: ${explicit_target_triple}")
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
  set(LIBC_TARGET_ARCHITECTURE_IS_ARM TRUE)
elseif(LIBC_TARGET_ARCHITECTURE STREQUAL "aarch64")
  set(LIBC_TARGET_ARCHITECTURE_IS_AARCH64 TRUE)
elseif(LIBC_TARGET_ARCHITECTURE STREQUAL "x86_64")
  set(LIBC_TARGET_ARCHITECTURE_IS_X86 TRUE)
elseif(LIBC_TARGET_ARCHITECTURE STREQUAL "riscv64")
  set(LIBC_TARGET_ARCHITECTURE_IS_RISCV64 TRUE)
  set(LIBC_TARGET_ARCHITECTURE "riscv")
elseif(LIBC_TARGET_ARCHITECTURE STREQUAL "riscv32")
  set(LIBC_TARGET_ARCHITECTURE_IS_RISCV32 TRUE)
  set(LIBC_TARGET_ARCHITECTURE "riscv")
elseif(LIBC_TARGET_ARCHITECTURE STREQUAL "amdgpu")
  set(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU TRUE)
elseif(LIBC_TARGET_ARCHITECTURE STREQUAL "nvptx")
  set(LIBC_TARGET_ARCHITECTURE_IS_NVPTX TRUE)
else()
  message(FATAL_ERROR
          "Unsupported libc target architecture ${LIBC_TARGET_ARCHITECTURE}")
endif()

if(LIBC_TARGET_OS STREQUAL "baremetal")
  set(LIBC_TARGET_OS_IS_BAREMETAL TRUE)
elseif(LIBC_TARGET_OS STREQUAL "linux")
  set(LIBC_TARGET_OS_IS_LINUX TRUE)
elseif(LIBC_TARGET_OS STREQUAL "poky" OR LIBC_TARGET_OS STREQUAL "suse")
  # poky are ustom Linux-base systems created by yocto. Since these are Linux
  # images, we change the LIBC_TARGET_OS to linux. This define is used to
  # include the right directories during compilation.
  #
  # openSUSE uses different triple format which causes LIBC_TARGET_OS to be
  # computed as "suse" instead of "linux".
  set(LIBC_TARGET_OS_IS_LINUX TRUE)
  set(LIBC_TARGET_OS "linux")
elseif(LIBC_TARGET_OS STREQUAL "darwin")
  set(LIBC_TARGET_OS_IS_DARWIN TRUE)
elseif(LIBC_TARGET_OS STREQUAL "windows")
  set(LIBC_TARGET_OS_IS_WINDOWS TRUE)
elseif(LIBC_TARGET_OS STREQUAL "gpu")
  set(LIBC_TARGET_OS_IS_GPU TRUE)
else()
  message(FATAL_ERROR
          "Unsupported libc target operating system ${LIBC_TARGET_OS}")
endif()


# If the compiler target triple is not the same as the triple specified by
# LIBC_TARGET_TRIPLE or LLVM_RUNTIMES_TARGET, we will add a --target option
# if the compiler is clang. If the compiler is GCC we just error out as there
# is no equivalent of an option like --target.
if(explicit_target_triple AND
   (NOT (libc_compiler_triple STREQUAL explicit_target_triple)))
  set(LIBC_CROSSBUILD TRUE)
  if(CMAKE_COMPILER_IS_GNUCXX)
    message(FATAL_ERROR
            "GCC target triple (${libc_compiler_triple}) and the explicity "
            "specified target triple (${explicit_target_triple}) do not match.")
  else()
    list(APPEND
         LIBC_COMPILE_OPTIONS_DEFAULT "--target=${explicit_target_triple}")
  endif()
endif()

message(STATUS
        "Building libc for ${LIBC_TARGET_ARCHITECTURE} on ${LIBC_TARGET_OS}")
