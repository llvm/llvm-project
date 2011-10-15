# See docs/CMake.html for instructions about how to build Compiler-RT with CMake.

PROJECT( CompilerRT C )
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

set(PACKAGE_NAME compiler-rt)
set(PACKAGE_VERSION 1.0svn)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules )

# add definitions
include(DefineCompilerFlags)

# Disallow in-source build
INCLUDE( MacroEnsureOutOfSourceBuild )
MACRO_ENSURE_OUT_OF_SOURCE_BUILD(
 "${PROJECT_NAME} requires an out of source build. Please create a separate build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
 )

INCLUDE( ${CMAKE_SOURCE_DIR}/cmake/ConfigureChecks.cmake )
CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/cmake/config.h.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/config.h )

INCLUDE_DIRECTORIES(
 ${CMAKE_CURRENT_BINARY_DIR}
)

SET( Achitectures
 i386 x86_64 ppc arm
 )

SET( Configurations
 Debug Release Profile
 )

# Only build Blocks Runtime if the compiler has enough support
IF( WIN32 OR MSVC OR HAVE_OSATOMIC_COMPARE_AND_SWAP_INT OR HAVE_SYNC_BOOL_COMPARE_AND_SWAP_INT )
  SET(BUILD_BLOCKS_RUNTIME TRUE)
ELSE( WIN32 OR MSVC OR HAVE_OSATOMIC_COMPARE_AND_SWAP_INT OR HAVE_SYNC_BOOL_COMPARE_AND_SWAP_INT )
  SET(BUILD_BLOCKS_RUNTIME FALSE)
ENDIF( WIN32 OR MSVC OR HAVE_OSATOMIC_COMPARE_AND_SWAP_INT OR HAVE_SYNC_BOOL_COMPARE_AND_SWAP_INT )

IF( BUILD_BLOCKS_RUNTIME )
  ADD_SUBDIRECTORY( BlocksRuntime )
ELSE( BUILD_BLOCKS_RUNTIME )
  MESSAGE(STATUS "No suitable atomic operation routines detected, skipping Blocks Runtime")
ENDIF( BUILD_BLOCKS_RUNTIME )

ADD_SUBDIRECTORY( lib )

# Enable Test Suit:
INCLUDE( MacroAddCheckTest )
ADD_SUBDIRECTORY( test )
