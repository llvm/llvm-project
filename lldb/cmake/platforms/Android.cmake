# Toolchain config for Android standalone NDK.
#
# Usage:
# build host llvm and clang first
# cmake -DCMAKE_TOOLCHAIN_FILE=../lldb/cmake/platforms/Android.cmake \
#       -DANDROID_TOOLCHAIN_DIR=<toolchain_dir> \
#       -DANDROID_ABI=<target_abi> \
#       -DCMAKE_CXX_COMPILER_VERSION=<gcc_version> \
#       -DLLVM_TARGET_ARCH=<llvm_target_arch> \
#       -DLLVM_TARGETS_TO_BUILD=<llvm_targets_to_build> \
#       -DLLVM_TABLEGEN=<path_to_llvm-tblgen> \
#       -DCLANG_TABLEGEN=<path_to_clang-tblgen>
#
# Current Support:
#   ANDROID_ABI = x86, x86_64
#   CMAKE_CXX_COMPILER_VERSION = 4.9
#   LLVM_TARGET_ARCH = X86
#   LLVM_TARGETS_TO_BUILD = X86
#   LLVM_TABLEGEN = path to host llvm-tblgen
#   CLANG_TABLEGEN = path to host clang-tblgen

if( DEFINED CMAKE_CROSSCOMPILING )
 return()
endif()

get_property( IS_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE )
if( IS_IN_TRY_COMPILE )
 # this seems necessary and works fine but I'm unsure if it breaks anything
 return()
endif()

set( CMAKE_SYSTEM_NAME Linux )
include( CMakeForceCompiler )

# flags and definitions
remove_definitions( -DANDROID -D__ANDROID__ )
add_definitions( -DANDROID -D__ANDROID_NDK__ -DLLDB_DISABLE_LIBEDIT )
set( ANDROID True )
set( __ANDROID_NDK__ True )
set( LLDB_DEFAULT_DISABLE_LIBEDIT True )

# linking lldb-server statically for Android avoids the need to ship two
# binaries (pie for API 21+ and non-pie for API 16-). It's possible to use
# a non-pie shim on API 16-, but that requires lldb-server to dynamically export
# its symbols, which significantly increases the binary size. Static linking, on
# the other hand, has little to no effect on the binary size.
if( NOT DEFINED LLVM_BUILD_STATIC )
 set( LLVM_BUILD_STATIC True  CACHE INTERNAL "" FORCE )
 set( LLVM_ENABLE_PIC   FALSE CACHE INTERNAL "" FORCE )
 set( BUILD_SHARED_LIBS FALSE CACHE INTERNAL "" FORCE )
endif()

set( ANDROID_ABI "${ANDROID_ABI}" CACHE INTERNAL "Android Abi" FORCE )
if( ANDROID_ABI STREQUAL "x86" )
 set( CMAKE_SYSTEM_PROCESSOR "i686" )
 set( ANDROID_TOOLCHAIN_NAME "i686-linux-android" )
elseif( ANDROID_ABI STREQUAL "x86_64" )
 set( CMAKE_SYSTEM_PROCESSOR "x86_64" )
 set( ANDROID_TOOLCHAIN_NAME "x86_64-linux-android" )
elseif( ANDROID_ABI STREQUAL "armeabi" )
 set( CMAKE_SYSTEM_PROCESSOR "armv5te" )
 set( ANDROID_TOOLCHAIN_NAME "arm-linux-androideabi" )
elseif( ANDROID_ABI STREQUAL "aarch64" )
 set( CMAKE_SYSTEM_PROCESSOR "aarch64" )
 set( ANDROID_TOOLCHAIN_NAME "aarch64-linux-android" )
elseif( ANDROID_ABI STREQUAL "mips" )
 set( CMAKE_SYSTEM_PROCESSOR "mips" )
 set( ANDROID_TOOLCHAIN_NAME "mipsel-linux-android" )
elseif( ANDROID_ABI STREQUAL "mips64" )
 set( CMAKE_SYSTEM_PROCESSOR "mips64" )
 set( ANDROID_TOOLCHAIN_NAME "mips64el-linux-android" )
else()
 message( SEND_ERROR "Unknown ANDROID_ABI = \"${ANDROID_ABI}\"." )
endif()

set( ANDROID_TOOLCHAIN_DIR "${ANDROID_TOOLCHAIN_DIR}" CACHE PATH "Android standalone toolchain directory" )
set( ANDROID_SYSROOT "${ANDROID_TOOLCHAIN_DIR}/sysroot" CACHE PATH "Android Sysroot" )

# CMAKE_EXECUTABLE_SUFFIX is undefined in CMAKE_TOOLCHAIN_FILE
if( WIN32 )
 set( EXECUTABLE_SUFFIX ".exe" )
endif()

set( PYTHON_EXECUTABLE "${ANDROID_TOOLCHAIN_DIR}/bin/python${EXECUTABLE_SUFFIX}" CACHE PATH "Python exec path" )

if( NOT CMAKE_C_COMPILER )
 set( CMAKE_C_COMPILER   "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-gcc${EXECUTABLE_SUFFIX}"     CACHE PATH "C compiler" )
 set( CMAKE_CXX_COMPILER "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-g++${EXECUTABLE_SUFFIX}"     CACHE PATH "C++ compiler" )
 set( CMAKE_ASM_COMPILER "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-gcc${EXECUTABLE_SUFFIX}"     CACHE PATH "assembler" )
 set( CMAKE_STRIP        "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-strip${EXECUTABLE_SUFFIX}"   CACHE PATH "strip" )
 set( CMAKE_AR           "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-ar${EXECUTABLE_SUFFIX}"      CACHE PATH "archive" )
 set( CMAKE_LINKER       "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-ld${EXECUTABLE_SUFFIX}"      CACHE PATH "linker" )
 set( CMAKE_NM           "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-nm${EXECUTABLE_SUFFIX}"      CACHE PATH "nm" )
 set( CMAKE_OBJCOPY      "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-objcopy${EXECUTABLE_SUFFIX}" CACHE PATH "objcopy" )
 set( CMAKE_OBJDUMP      "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-objdump${EXECUTABLE_SUFFIX}" CACHE PATH "objdump" )
 set( CMAKE_RANLIB       "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-ranlib${EXECUTABLE_SUFFIX}"  CACHE PATH "ranlib" )
endif()

set( ANDROID_CXX_FLAGS "--sysroot=${ANDROID_SYSROOT} -funwind-tables -fsigned-char -no-canonical-prefixes" )
# TODO: different ARM abi have different flags such as neon, vfpv etc
if( X86 )
 set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -funswitch-loops -finline-limit=300" )
elseif( ANDROID_ABI STREQUAL "armeabi" )
 # 64 bit atomic operations used in c++ libraries require armv7-a instructions
 # armv5te and armv6 were tried but do not work.
 set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -march=armv7-a -mthumb" )
 if( LLVM_BUILD_STATIC )
  # Temporary workaround for static linking with the latest API.
  set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -DANDROID_ARM_BUILD_STATIC" )
 endif()
elseif( ANDROID_ABI STREQUAL "mips" )
 # http://b.android.com/182094
 list( FIND LLDB_SYSTEM_LIBS atomic index )
 if( index EQUAL -1 )
  list( APPEND LLDB_SYSTEM_LIBS atomic )
  set( LLDB_SYSTEM_LIBS ${LLDB_SYSTEM_LIBS} CACHE INTERNAL "" FORCE )
 endif()
 if( LLVM_BUILD_STATIC )
  # Temporary workaround for static linking with the latest API.
  set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -DANDROID_MIPS_BUILD_STATIC" )
 endif()
endif()

if( NOT LLVM_BUILD_STATIC )
 # PIE is required for API 21+ so we enable it if we're not statically linking
 # unfortunately, it is not supported before API 16 so we need to do something
 # else there see http://llvm.org/pr23457
 set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -pie -fPIE" )
endif()

# linker flags
set( ANDROID_CXX_FLAGS    "${ANDROID_CXX_FLAGS} -fdata-sections -ffunction-sections" )
set( ANDROID_LINKER_FLAGS "${ANDROID_LINKER_FLAGS} -Wl,--gc-sections" )

# cache flags
set( CMAKE_CXX_FLAGS           ""                        CACHE STRING "c++ flags" )
set( CMAKE_C_FLAGS             ""                        CACHE STRING "c flags" )
set( CMAKE_EXE_LINKER_FLAGS    "-Wl,-z,nocopyreloc"      CACHE STRING "executable linker flags" )
set( ANDROID_CXX_FLAGS         "${ANDROID_CXX_FLAGS}"    CACHE INTERNAL "Android c/c++ flags" )
set( ANDROID_LINKER_FLAGS      "${ANDROID_LINKER_FLAGS}" CACHE INTERNAL "Android c/c++ linker flags" )

# final flags
set( CMAKE_CXX_FLAGS           "${ANDROID_CXX_FLAGS} ${CMAKE_CXX_FLAGS}" )
set( CMAKE_C_FLAGS             "${ANDROID_CXX_FLAGS} ${CMAKE_C_FLAGS}" )
set( CMAKE_EXE_LINKER_FLAGS    "${ANDROID_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}" )

# global includes and link directories
set( ANDROID_INCLUDE_DIRS "${ANDROID_TOOLCHAIN_DIR}/include/c++/${ANDROID_COMPILER_VERSION}" )
list( APPEND ANDROID_INCLUDE_DIRS "${ANDROID_TOOLCHAIN_DIR}/include/python2.7" )
include_directories( SYSTEM "${ANDROID_SYSROOT}/usr/include" ${ANDROID_INCLUDE_DIRS} )

# target environment
set( CMAKE_FIND_ROOT_PATH "${ANDROID_TOOLCHAIN_DIR}/bin" "${ANDROID_TOOLCHAIN_DIR}/${ANDROID_TOOLCHAIN_NAME}" "${ANDROID_SYSROOT}" )

# only search for libraries and includes in the ndk toolchain
set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

################# BEGIN EVIL HACK ##################
# lldb-server links against libdl even though it's not being used and
# libdl.a is currently missing from the toolchain (b.android.com/178517).
# Therefore, in order to statically link lldb-server, we need a temporary
# workaround. This creates a dummy libdl.a stub until the actual
# libdl.a can be implemented in the toolchain.
if( LLVM_BUILD_STATIC )
 set( libdl "${CMAKE_BINARY_DIR}/libdl_stub" )
 file( MAKE_DIRECTORY ${libdl} )
 file( WRITE "${libdl}/libdl.c" "
#include <dlfcn.h>
void *       dlopen  (const char *filename, int flag)   { return 0; }
const char * dlerror (void)                             { return 0; }
void *       dlsym   (void *handle, const char *symbol) { return 0; }
int          dlclose (void *handle)                     { return 0; }")
 set( flags "${CMAKE_C_FLAGS}" )
 separate_arguments( flags )
 execute_process( COMMAND ${CMAKE_C_COMPILER} ${flags} -c ${libdl}/libdl.c -o ${libdl}/libdl.o )
 execute_process( COMMAND ${CMAKE_AR} rcs ${libdl}/libdl.a ${libdl}/libdl.o )
 set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${libdl}" )
endif()
################# END EVIL HACK ##################
