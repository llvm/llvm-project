# This file sets up a CMakeCache for building the clang for Fuchsia.

set(LLVM_TARGETS_TO_BUILD X86;ARM;AArch64;RISCV CACHE STRING "")

set(PACKAGE_VENDOR Fuchsia CACHE STRING "")

set(_FUCHSIA_ENABLE_PROJECTS "bolt;clang;clang-tools-extra;lld;llvm;polly")
set(LLVM_ENABLE_RUNTIMES "compiler-rt;libcxx;libcxxabi;libunwind" CACHE STRING "")

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
if (APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "")
elseif(WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "")
endif()

# Setup toolchain.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  dsymutil
  llvm-ar
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

if (LLVM_BUILTIN_TARGETS OR LLVM_RUNTIME_TARGETS)
  set(_FUCHSIA_RUNTIMES_TOOLS
    builtins
    runtimes)
endif()

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
  ${_FUCHSIA_RUNTIMES_TOOLS}
  ${LLVM_TOOLCHAIN_TOOLS}
  CACHE STRING "")

set(_FUCHSIA_DISTRIBUTIONS Toolchain)

set(LLVM_DISTRIBUTIONS ${_FUCHSIA_DISTRIBUTIONS} CACHE STRING "")
set(LLVM_ENABLE_PROJECTS ${_FUCHSIA_ENABLE_PROJECTS} CACHE STRING "")
