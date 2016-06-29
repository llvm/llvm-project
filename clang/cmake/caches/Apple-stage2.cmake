# This file sets up a CMakeCache for Apple-style stage2 bootstrap. It is
# specified by the stage1 build.

set(LLVM_TARGETS_TO_BUILD X86 ARM AArch64 CACHE STRING "") 
set(PACKAGE_VENDOR Apple CACHE STRING "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_TOOL_CLANG_TOOLS_EXTRA_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_SCAN_BUILD_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_SCAN_VIEW_BUILD OFF CACHE BOOL "")
set(CLANG_LINKS_TO_CREATE clang++ cc c++ CACHE STRING "")
set(CMAKE_MACOSX_RPATH ON CACHE BOOL "")
set(LLVM_ENABLE_ZLIB ON CACHE BOOL "")
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
set(LLVM_EXTERNALIZE_DEBUGINFO ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT OFF CACHE BOOL "")
set(BUG_REPORT_URL "http://developer.apple.com/bugreporter/" CACHE STRING "")

set(LLVM_BUILD_EXTERNAL_COMPILER_RT ON CACHE BOOL "Build Compiler-RT with just-built clang")
set(COMPILER_RT_ENABLE_IOS ON CACHE BOOL "Build iOS Compiler-RT libraries")

# Make unit tests (if present) part of the ALL target
set(LLVM_BUILD_TESTS ON CACHE BOOL "")

set(LLVM_ENABLE_LTO ON CACHE BOOL "")
set(CMAKE_C_FLAGS "-fno-stack-protector -fno-common -Wno-profile-instr-unprofiled" CACHE STRING "")
set(CMAKE_CXX_FLAGS "-fno-stack-protector -fno-common -Wno-profile-instr-unprofiled" CACHE STRING "")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -gline-tables-only -DNDEBUG" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -gline-tables-only -DNDEBUG" CACHE STRING "")
set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")

set(LIBCXX_INSTALL_LIBRARY OFF CACHE BOOL "")
set(LIBCXX_INSTALL_HEADERS ON CACHE BOOL "")
set(LIBCXX_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_LTO_VERSION_OFFSET 3000 CACHE STRING "")

# setup toolchain
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  llvm-dsymutil
  llvm-cov
  llvm-dwarfdump
  llvm-profdata
  llvm-objdump
  llvm-nm
  llvm-size
  CACHE STRING "")

set(LLVM_DISTRIBUTION_COMPONENTS
  clang
  LTO
  clang-format
  clang-headers
  libcxx-headers
  ${LLVM_TOOLCHAIN_TOOLS}
  CACHE STRING "")

# test args

set(LLVM_LIT_ARGS "--xunit-xml-output=testresults.xunit.xml -v" CACHE STRING "")
