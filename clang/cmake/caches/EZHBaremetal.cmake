set(LLVM_TARGETS_TO_BUILD X86;EZH CACHE STRING "")

set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES "compiler-rt;libc" CACHE STRING "")

set(LLVM_BUILTIN_TARGETS "ezh-none-elf" CACHE STRING "")
set(LLVM_RUNTIME_TARGETS "ezh-none-elf" CACHE STRING "")

set(BUILTINS_ezh-none-elf_CMAKE_SYSTEM_NAME Generic CACHE STRING "")
set(BUILTINS_ezh-none-elf_COMPILER_RT_BAREMETAL_BUILD ON CACHE BOOL "")
set(BUILTINS_ezh-none-elf_COMPILER_RT_OS_DIR "baremetal" CACHE STRING "")

set(RUNTIMES_ezh-none-elf_LLVM_LIBC_FULL_BUILD ON CACHE BOOL "")
set(RUNTIMES_ezh-none-elf_LIBC_TARGET_OS "baremetal" CACHE STRING "")
set(RUNTIMES_ezh-none-elf_LIBC_TARGET_ARCHITECTURE "ezh" CACHE STRING "")
set(RUNTIMES_ezh-none-elf_BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(RUNTIMES_ezh-none-elf_CMAKE_C_FLAGS "-Wno-everything -nostdlib" CACHE STRING "")

set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  llc
  llvm-ar
  llvm-nm
  llvm-objdump
  llvm-ranlib
  llvm-readobj
  llvm-size
  opt
  CACHE STRING "")

set(LLVM_DISTRIBUTION_COMPONENTS
  clang
  lld
  clang-resource-headers
  builtins-ezh-none-elf
  runtimes
  ${LLVM_TOOLCHAIN_TOOLS}
  CACHE STRING "")
