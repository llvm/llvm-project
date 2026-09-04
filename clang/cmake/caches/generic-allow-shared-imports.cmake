# Override Platform/Generic.cmake's TARGET_SUPPORTS_SHARED_LIBS=FALSE.
#
# The baremetal builtins ExternalProject (CMAKE_SYSTEM_NAME=Generic) does
# find_package(LLVM) which loads LLVMExports.cmake.  That file declares
# host-only shared libraries (LTO, Remarks) as SHARED IMPORTED targets.
# CMake 4.x rejects these on platforms where TARGET_SUPPORTS_SHARED_LIBS
# is FALSE, even though the builtins build never links against them.
#
# Loaded via CMAKE_PROJECT_INCLUDE so it runs after project() (which loads
# Platform/Generic.cmake) but before find_package(LLVM).
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
