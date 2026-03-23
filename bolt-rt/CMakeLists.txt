cmake_minimum_required(VERSION 3.20.0)

include(CheckCXXCompilerFlag)
include(CheckIncludeFiles)
include(CMakeParseArguments)
include(GNUInstallDirs)

# Is this built via -DLLVM_ENABLE_RUNTIMES="bolt-rt"?.
if(LLVM_RUNTIMES_BUILD)
  set(LLVM_SUBPROJECT_TITLE "Bolt-RT")
# Is this being built directly?
elseif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(bolt-rt)
endif()

# Use LLVM_RUNTIMES_TARGET when built under runtimes.
if(LLVM_RUNTIMES_BUILD)
  set(BOLT_RT_TARGET_TRIPLE "${LLVM_RUNTIMES_TARGET}")
endif()

if(NOT BOLT_RT_TARGET_TRIPLE)
  message(STATUS "bolt_rt: BOLT_RT_TARGET_TRIPLE is unset, building for the host.")
  get_host_triple(BOLT_RT_TARGET_TRIPLE)
endif()

# Append --target for clang compilers. Otherwise it is assumed that
# one of the following has been configured.
# 1. CMAKE_CXX_FLAGS for stand alone builds.
# 2. RUNTIMES_<target-triple>_CMAKE_CXX_FLAGS for runtime builds.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND BOLT_RT_FLAGS
       "--target=${BOLT_RT_TARGET_TRIPLE}")
endif()

message(STATUS "bolt-rt: targeting ${BOLT_RT_TARGET_TRIPLE}")

# Add custom target for runtime builds.
if(LLVM_RUNTIMES_BUILD)
  add_custom_target(bolt-rt)
  set_target_properties(bolt-rt PROPERTIES FOLDER "BOLT-RT/Metatargets")
  add_llvm_install_targets(install-bolt-rt
                           DEPENDS bolt-rt
                           COMPONENTS bolt-rt)
endif()

# For runtime builds, the installation directory is already known,
# otherwise assume lib.
if(LLVM_RUNTIMES_BUILD)
  set(BOLT_RT_LIBRARY_DIR "${LLVM_LIBRARY_DIR}")
else()
  set(BOLT_RT_LIBRARY_DIR "lib${LLVM_LIBDIR_SUFFIX}")
endif()

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
  string(APPEND BOLT_RT_LIBRARY_DIR "/${BOLT_RT_TARGET_TRIPLE}")
  set(BOLT_RT_ARCHIVE_SUFFIX "")
else()
  # TODO(casey.smalley@arm.com): Does this need to account for
  # LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF?
  message(FATAL_ERROR "bolt_rt: Not implemented.")
endif()

# Build flags.
list(APPEND BOLT_RT_FLAGS
    -ffreestanding
    -fno-exceptions
    -fno-rtti
    -fno-stack-protector
    -fPIC
    # Runtime currently assumes omitted frame pointers for functions marked __attribute((naked)).
    # Protect against distros adding -fno-omit-frame-pointer and compiling with GCC.
    # Refs: llvm/llvm-project#148595 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77882
    -fomit-frame-pointer)

if(BOLT_RT_TARGET_TRIPLE MATCHES "x86_64")
  list(APPEND BOLT_RT_FLAGS
       -mno-sse
       -mgeneral-regs-only)
elseif(BOLT_RT_TARGET_TRIPLE MATCHES "aarch64")
  check_cxx_compiler_flag("-mno-outline-atomics" CXX_SUPPORTS_OUTLINE_ATOMICS)

  if(CXX_SUPPORTS_OUTLINE_ATOMICS)
    list(APPEND BOLT_RT_FLAGS
         -mno-outline-atomics
         -mgeneral-regs-only)
  endif()
endif()

# Build libraries.
check_include_files(elf.h HAVE_ELF_H)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/config.h")

function(add_bolt_rt_library)
  cmake_parse_arguments(ARG "" "TARGET" "SOURCES;FLAGS" ${ARGN})
  add_library(${ARG_TARGET} ${ARG_SOURCES})

  # Don't let the compiler think it can create calls to standard libs
  target_include_directories(${ARG_TARGET}
                             PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

  target_compile_options(${ARG_TARGET}
                         PRIVATE ${BOLT_RT_FLAGS} ${ARG_FLAGS})

  # Ouput archives into a per-triple path or with a per triple suffix.
  set_target_properties(${ARG_TARGET} PROPERTIES
                        ARCHIVE_OUTPUT_DIRECTORY "${BOLT_RT_LIBRARY_DIR}"
                        ARCHIVE_OUTPUT_NAME "${ARG_TARGET}${BOLT_RT_ARCHIVE_SUFFIX}")

  if(LLVM_RUNTIMES_BUILD)
    add_dependencies(bolt-rt ${ARG_TARGET})
  endif()

  # Install to a matching directory.
  install(TARGETS ${ARG_TARGET}
          DESTINATION "${BOLT_RT_LIBRARY_DIR}")
endfunction()

add_bolt_rt_library(TARGET bolt_rt_instr
                    SOURCES instr.cpp)

add_bolt_rt_library(TARGET bolt_rt_hugify
                    SOURCES hugify.cpp)
