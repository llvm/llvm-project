# Second stage of PGO (used by PGO-stage2-instrumented.cmake)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "clang;flang;lld" CACHE STRING "")
