# This file sets up a CMakeCache for building a single stage of a Fuchsia toolchain.
# It can be used to just build LLVM+Clang without building any runtimes
# Or LLVM+Clang+runtimes.
set(LLVM_TARGETS_TO_BUILD X86;ARM;AArch64;RISCV CACHE STRING "")

set(PACKAGE_VENDOR Fuchsia CACHE STRING "")

set(_FUCHSIA_ENABLE_PROJECTS "bolt;clang;clang-tools-extra;lld;llvm;polly")

set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
set(LLVM_ENABLE_DIA_SDK OFF CACHE BOOL "")
set(LLVM_ENABLE_FATLTO ON CACHE BOOL "")
set(LLVM_ENABLE_HTTPLIB ON CACHE BOOL "")
set(LLVM_ENABLE_LIBCXX ON CACHE BOOL "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_LLD ON CACHE BOOL "")
set(LLVM_ENABLE_LTO ON CACHE BOOL "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE BOOL "")
set(LLVM_ENABLE_PLUGINS OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")
set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB ON CACHE BOOL "")
set(LLVM_FORCE_BUILD_RUNTIME ON CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_STATIC_LINK_CXX_STDLIB ON CACHE BOOL "")
set(LLVM_USE_RELATIVE_PATHS_IN_FILES ON CACHE BOOL "")
set(LLDB_ENABLE_CURSES OFF CACHE BOOL "")
set(LLDB_ENABLE_LIBEDIT OFF CACHE BOOL "")

if(WIN32)
  set(FUCHSIA_DISABLE_DRIVER_BUILD ON)
endif()

if (NOT FUCHSIA_DISABLE_DRIVER_BUILD)
  set(LLVM_TOOL_LLVM_DRIVER_BUILD ON CACHE BOOL "")
  set(LLVM_DRIVER_TARGET llvm-driver)
endif()

set(CLANG_DEFAULT_CXX_STDLIB libc++ CACHE STRING "")
set(CLANG_DEFAULT_LINKER lld CACHE STRING "")
set(CLANG_DEFAULT_OBJCOPY llvm-objcopy CACHE STRING "")
set(CLANG_DEFAULT_RTLIB compiler-rt CACHE STRING "")
set(CLANG_DEFAULT_UNWINDLIB libunwind CACHE STRING "")
set(CLANG_ENABLE_ARCMT OFF CACHE BOOL "")
set(CLANG_ENABLE_STATIC_ANALYZER ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT OFF CACHE BOOL "")

set(ENABLE_LINKER_BUILD_ID ON CACHE BOOL "")
set(ENABLE_X86_RELAX_RELOCATIONS ON CACHE BOOL "")

# TODO(#67176): relative-vtables doesn't play well with different default
# visibilities. Making everything hidden visibility causes other complications
# let's choose default visibility for our entire toolchain.
set(CMAKE_C_VISIBILITY_PRESET default CACHE STRING "")
set(CMAKE_CXX_VISIBILITY_PRESET default CACHE STRING "")

set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# Setup the runtimes for Fuchsia toolchain.
set(FUCHSIA_TOOLCHAIN_RUNTIME_PLATFORMS "" CACHE STRING "")
LIST(LENGTH FUCHSIA_TOOLCHAIN_RUNTIME_PLATFORMS platform_count)
if (NOT platform_count EQUAL 0)
    set(FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_MODE "END2END" CACHE STRING "")
    set(LLVM_ENABLE_RUNTIMES "compiler-rt;libcxx;libcxxabi;libunwind" CACHE STRING "")
    set(FUCHSIA_TOOLCHAIN_RUNTIME_BUILTIN_TARGETS "" CACHE STRING "")
    set(FUCHSIA_TOOLCHAIN_RUNTIME_RUNTIME_TARGETS "" CACHE STRING "")
    set(FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_ID_LINK "" CACHE STRING "")
    set(FUCHSIA_RUNTIME_COMPONENTS "builtins;runtimes" CACHE STRING "")
endif()

foreach(platform IN LISTS FUCHSIA_TOOLCHAIN_RUNTIME_PLATFORMS)
    include(${CMAKE_CURRENT_LIST_DIR}/../../../runtimes/cmake/fuchsia/${platform}.cmake)
endforeach()


# Setup toolchain.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  dsymutil
  llvm-ar
  llvm-config
  llvm-cov
  llvm-cxxfilt
  llvm-debuginfod
  llvm-debuginfod-find
  llvm-dlltool
  ${LLVM_DRIVER_TARGET}
  llvm-dwarfdump
  llvm-dwp
  llvm-ifs
  llvm-gsymutil
  llvm-lib
  llvm-libtool-darwin
  llvm-lipo
  llvm-ml
  llvm-mt
  llvm-nm
  llvm-objcopy
  llvm-objdump
  llvm-otool
  llvm-pdbutil
  llvm-profdata
  llvm-rc
  llvm-ranlib
  llvm-readelf
  llvm-readobj
  llvm-size
  llvm-strings
  llvm-strip
  llvm-symbolizer
  llvm-undname
  llvm-xray
  opt-viewer
  sancov
  scan-build-py
  CACHE STRING "")

set(LLVM_Toolchain_DISTRIBUTION_COMPONENTS
  bolt
  clang
  lld
  clang-apply-replacements
  clang-doc
  clang-format
  clang-resource-headers
  clang-include-fixer
  clang-refactor
  clang-scan-deps
  clang-tidy
  clangd
  find-all-symbols
  ${FUCHSIA_RUNTIME_COMPONENTS}
  ${LLVM_TOOLCHAIN_TOOLS}
  CACHE STRING "")

if (NOT platform_count EQUAL 0)
  set(LLVM_BUILTIN_TARGETS "${FUCHSIA_TOOLCHAIN_RUNTIME_BUILTIN_TARGETS}" CACHE STRING "")
  set(LLVM_RUNTIME_TARGETS "${FUCHSIA_TOOLCHAIN_RUNTIME_RUNTIME_TARGETS}" CACHE STRING "")
  set(RUNTIME_BUILD_ID_LINK "${FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_ID_LINK}" CACHE STRING "")
endif()

set(_FUCHSIA_DISTRIBUTIONS Toolchain)

set(LLVM_DISTRIBUTIONS ${_FUCHSIA_DISTRIBUTIONS} CACHE STRING "")
set(LLVM_ENABLE_PROJECTS ${_FUCHSIA_ENABLE_PROJECTS} CACHE STRING "")
