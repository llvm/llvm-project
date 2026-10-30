include(${CMAKE_CURRENT_LIST_DIR}/clang.cmake)

# On Darwin, Clang links against the compiler-rt builtins shipped in the toolchain,
# so we must build them here.
set(LLVM_ENABLE_RUNTIMES "compiler-rt" CACHE STRING "")
set(LLVM_DISTRIBUTION_COMPONENTS ${LLVM_DISTRIBUTION_COMPONENTS} builtins CACHE STRING "" FORCE)
