# LLVM dylib settings for distribution cross-builds
#
# Loaded after hexagon-unknown-linux-musl-clang-dist.cmake to override the
# static linking defaults.  ELD is incompatible with dylib (libLW.so
# conflicts with libLLVM.so), so ELD must be excluded when this cache
# is used.

set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "" FORCE)
set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "" FORCE)
set(CLANG_LINK_LLVM_DYLIB ON CACHE BOOL "" FORCE)
