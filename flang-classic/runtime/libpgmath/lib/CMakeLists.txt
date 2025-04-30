#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# This value will be the same as LIBPGMATH_SYSTEM_PROCESSOR
set(PROCESSOR ${LIBPGMATH_SYSTEM_PROCESSOR})
if(${LIBPGMATH_WITH_GENERIC} OR (NOT ${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "x86_64|aarch64"))
  set(PROCESSOR "generic")
endif()

find_program(SH_PROGRAM NAMES sh bash)

set_property(GLOBAL APPEND PROPERTY "TARGET_OBJECTS")

if(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "x86_64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")

    set(DEFINITIONS_L1
      HOST_LINUX LINUX LINUX86 LINUX8664 MAXCPUS=256 MAXCPUSL=8 MAXCPUSR=8
      TARGET_LINUX TARGET_LINUX_X86 TARGET_LINUX_X8664 TARGET_X86 TARGET_X8664
      __gnu_linux__ PG_PIC)
    set(FLAGS_L1 "-m64 -O3 ")

    set(DEFINITIONS_L2
      LINUX LINUX86 LINUX8664 MAXCPUS=256 MAXCPUSL=8 MAXCPUSR=8 __gnu_linux__
      TARGET_LINUX TARGET_LINUX_X86 TARGET_LINUX_X8664 TARGET_X86 TARGET_X8664
      PG_PIC)
    set(FLAGS_L2 "-m64 -O3 -mtune=core-avx2 -march=core-avx2 ")

    # common
    # Definitions and compiler flags for level 1 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L1" ${DEFINITIONS_L1})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L1" ${FLAGS_L1})

    # Definitions and compiler flags for level 2 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L2" ${DEFINITIONS_L2})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L2" ${FLAGS_L2})
    # common

    # x86_64
    # Definitions and compiler flags for level 1 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_X8664_L1" ${DEFINITIONS_L1})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_X8664_L1" ${FLAGS_L1})

    # Definitions and compiler flags for level 2 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_X8664_L2" ${DEFINITIONS_L2})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_X8664_L2" ${FLAGS_L2})
    # x86_64

  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin")
    set(DEFINITIONS_L1 HOST_OSX MAXCPUS=256 MAXCPUSL=8 MAXCPUSR=4 OSX OSX86
      OSX8664 TARGET_OSX TARGET_OSX_X86 TARGET_OSX_X8664 TARGET_X86
      TARGET_X8664 __gnu_osx__ PG_PIC)
    set(FLAGS_L1 "-m64 -O3 -fomit-frame-pointer ")

    set(DEFINITIONS_L2 OSX86 TARGET_OSX TARGET_OSX_X86 TARGET_OSX_X8664 PG_PIC TARGET_X86 TARGET_X8664)
    set(FLAGS_L2 "-m64 -O3 -fomit-frame-pointer -mavx2 -mfma ")

    # common
    # Definitions and compiler flags for level 1 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L1" ${DEFINITIONS_L1})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L1" ${FLAGS_L1})

    # Definitions and compiler flags for level 2 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L2" ${DEFINITIONS_L2})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L2" ${FLAGS_L2})
    # common

    # x86_64
    # Definitions and compiler flags for level 1 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_X8664_L1" ${DEFINITIONS_L1})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_X8664_L1" ${FLAGS_L1})

    # Definitions and compiler flags for level 2 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_X8664_L2" ${DEFINITIONS_L2})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_X8664_L2" ${FLAGS_L2})
    # x86_64
  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Windows")
    set(DEFINITIONS_L1
      TARGET_WIN_X8664 TARGET_WIN _PGI_NOBUILTINS MAXCPUS=256
      MAXCPUSL=8 MAXCPUSR=8 WINNT WIN64 LONG_IS_32
      HOST_WIN TARGET_X8664)
#problems with -O2 and higher using vectorcall ABI
#    set(FLAGS_L1 "-m64 -O2 ")
    set(FLAGS_L2 "-m64 ")

    set(DEFINITIONS_L2
      TARGET_WIN_X8664 TARGET_WIN _PGI_NOBUILTINS MAXCPUS=256
      MAXCPUSL=8 MAXCPUSR=8 WINNT WIN64 LONG_IS_32
      HOST_WIN TARGET_X8664)
#problems with -O2 and higher using vectorcall ABI
#    set(FLAGS_L2 "-m64 -O2 ")
    set(FLAGS_L2 "-m64 ")

    # common
    # Definitions and compiler flags for level 1 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L1" ${DEFINITIONS_L1})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L1" ${FLAGS_L1})

    # Definitions and compiler flags for level 2 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L2" ${DEFINITIONS_L2})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L2" ${FLAGS_L2})
    # common

    # x86_64
    # Definitions and compiler flags for level 1 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_X8664_L1" ${DEFINITIONS_L1})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_X8664_L1" ${FLAGS_L1})

    # Definitions and compiler flags for level 2 directories
    set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_X8664_L2" ${DEFINITIONS_L2})
    set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_X8664_L2" ${FLAGS_L2})
    # x86_64
  else()
    message(FATAL "Operating System not supported.")
  endif()
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "ppc64le" AND NOT ${LIBPGMATH_WITH_GENERIC})

  set(DEFINITIONS_L1 HOST_LINUX MAXCPUS=256 MAXCPUSL=8 MAXCPUSR=8 TARGET_LINUX
    TARGET_LINUX_POWER TARGET_LLVM TARGET_LLVM_LINUXPOWER TM_I8 PG_PIC)
  set(FLAGS_L1 "-m64 -O3 ")

  set(DEFINITIONS_L2 HOST_LINUX MAXCPUS=256 MAXCPUSL=8 MAXCPUSR=8 TARGET_LINUX
    TARGET_LINUX_POWER TARGET_LLVM TARGET_LLVM_LINUXPOWER TM_I8 PG_PIC)
  set(FLAGS_L2 "-m64 -O3 ")

  # common
  # Definitions and compiler flags for level 1 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L1" ${DEFINITIONS_L1})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L1" ${FLAGS_L1})

  # Definitions and compiler flags for level 2 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L2" ${DEFINITIONS_L2})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L2" ${FLAGS_L2})
  # common

  # ppc64le
  # Definitions and compiler flags for level 1 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_PPC64LE_L1" ${DEFINITIONS_L1})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_PPC64LE_L1" ${FLAGS_L1})

  # Definitions and compiler flags for level 2 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_PPC64LE_L2" ${DEFINITIONS_L2})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_PPC64LE_L2" ${FLAGS_L2})
  # ppc64le
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Windows")
    set(DEFINITIONS_L1 HOST_WIN MAXCPUS=256 MAXCPUSL=8 MAXCPUSR=8 TARGET_WIN
    TARGET_WIN_ARM64 WIN_ARM64 WINNT WIN64 TARGET_ARM64 _PGI_NOBUILTINS)
    if("${CMAKE_C_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC")
      set(FLAGS_L1 "/fp:fast")
    endif()

    set(DEFINITIONS_L2 HOST_WIN MAXCPUS=256 MAXCPUSL=8 MAXCPUSR=8 TARGET_WIN
      TARGET_WIN_ARM64 WIN_ARM64 WINNT WIN64 TARGET_ARM64 _PGI_NOBUILTINS)
    if("${CMAKE_C_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC")
      set(FLAGS_L2 "/fp:fast")
    endif()
  else()
    set(DEFINITIONS_L1 HOST_LINUX MAXCPUS=8192 MAXCPUSL=128 MAXCPUSR=128
      TARGET_LINUX TARGET_LINUX_ARM TARGET_LINUX_ARM64 TARGET_ARM64 TARGET_LLVM
      TARGET_LLVM_LINUXAARCH64 TM_I8)
    set(FLAGS_L1 "-O3 -mcpu=${LLVM_FLANG_CPU_TARGET} -ffast-math -funroll-loops ")
    set(FLAGS_L1 "${FLAGS_L1} -ffp-contract=fast")

    set(DEFINITIONS_L2 HOST_LINUX MAXCPUS=8192 MAXCPUSL=128 MAXCPUSR=128
      TARGET_LINUX TARGET_LINUX_ARM TARGET_LINUX_ARM64 TARGET_ARM64 TARGET_LLVM TM_I8)
    set(FLAGS_L2 "-O3 -mcpu=${LLVM_FLANG_CPU_TARGET} -ffast-math -funroll-loops -ffp-contract=fast")
  endif()

  # common
  # Definitions and compiler flags for level 1 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L1" ${DEFINITIONS_L1})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L1" ${FLAGS_L1})

  # Definitions and compiler flags for level 2 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L2" ${DEFINITIONS_L2})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L2" ${FLAGS_L2})
  # common

  # aarch64
  # Definitions and compiler flags for level 1 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_ARM64_L1" ${DEFINITIONS_L1})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_ARM64_L1" ${FLAGS_L1})

  # Definitions and compiler flags for level 2 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_ARM64_L2" ${DEFINITIONS_L2})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_ARM64_L2" ${FLAGS_L2})
  # aarch64
else()
  set(DEFINITIONS_L1 HOST_LINUX MAXCPUS=256 MAXCPUSL=8 MAXCPUSR=8 TARGET_LINUX
    TARGET_LINUX_64 TARGET_LLVM TARGET_LLVM_64 TARGET_LLVM_LINUX64
    TARGET_LINUX_GENERIC PG_PIC)
  set(FLAGS_L1 "-O3 -ffp-contract=fast ")

  set(DEFINITIONS_L2 HOST_LINUX MAXCPUS=256 MAXCPUSL=8 MAXCPUSR=8 TARGET_LINUX
    TARGET_LINUX_64 TARGET_LLVM TARGET_LLVM_64 TARGET_LLVM_LINUX64
    TARGET_LINUX_GENERIC PG_PIC)
  set(FLAGS_L2 "-O3 -ffp-contract=fast ")

  # common
  # Definitions and compiler flags for level 1 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L1" ${DEFINITIONS_L1})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L1" ${FLAGS_L1})

  # Definitions and compiler flags for level 2 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_L2" ${DEFINITIONS_L2})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_L2" ${FLAGS_L2})
  # common

  # generic
  # Definitions and compiler flags for level 1 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_GENERIC_L1" ${DEFINITIONS_L1})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_GENERIC_L1" ${FLAGS_L1})

  # Definitions and compiler flags for level 2 directories
  set_property(GLOBAL APPEND PROPERTY "DEFINITIONS_GENERIC_L2" ${DEFINITIONS_L2})
  set_property(GLOBAL APPEND_STRING PROPERTY "FLAGS_GENERIC_L2" ${FLAGS_L2})
  # generic
endif()

include_directories(common)
if(${PROCESSOR} MATCHES "x86_64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  include_directories(x86_64)
# elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "ppc64le" AND NOT ${LIBPGMATH_WITH_GENERIC})
#   include_directories(ppc64le)
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  include_directories(generic)
  include_directories(aarch64)
elseif(${PROCESSOR} MATCHES "ppc64le|generic" OR ${LIBPGMATH_WITH_GENERIC})
  include_directories(generic)
endif()

# Add directories to build
add_subdirectory(common)
if(${PROCESSOR} MATCHES "x86_64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  add_subdirectory(x86_64)
# elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "ppc64le" AND NOT ${LIBPGMATH_WITH_GENERIC})
#   add_subdirectory(ppc64le)
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  add_subdirectory(generic)
  add_subdirectory(aarch64)
elseif(${PROCESSOR} MATCHES "ppc64le|generic" OR ${LIBPGMATH_WITH_GENERIC})
  add_subdirectory(generic)
endif()

# Create libmath static library
get_property(TARGET_OBJECTS GLOBAL PROPERTY "TARGET_OBJECTS")

add_library(${LIBPGMATH_LIBRARY_NAME}_static STATIC ${TARGET_OBJECTS})
add_library(${LIBPGMATH_LIBRARY_NAME} SHARED ${TARGET_OBJECTS})
target_link_libraries(${LIBPGMATH_LIBRARY_NAME} INTERFACE ${LIBPGMATH_LIBRARY_NAME}_static)
if(${LIBPGMATH_WITH_GENERIC})
    target_link_libraries(${LIBPGMATH_LIBRARY_NAME} PRIVATE m)
endif()
if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Windows")
    set_target_properties(${LIBPGMATH_LIBRARY_NAME}_static PROPERTIES OUTPUT_NAME lib${LIBPGMATH_LIBRARY_NAME})
    set_target_properties(${LIBPGMATH_LIBRARY_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
if(${CMAKE_GENERATOR} MATCHES "MSYS Makefiles")
add_custom_command (TARGET ${LIBPGMATH_LIBRARY_NAME} PRE_LINK
    COMMAND  sed 's/ /\\n/g' CMakeFiles/pgmath.dir/objects1.rsp > objects1.txt
    COMMAND  ${CMAKE_COMMAND} -E __create_def libpgmath.def objects1.txt
)
endif()

else()
    set_target_properties(${LIBPGMATH_LIBRARY_NAME}_static PROPERTIES OUTPUT_NAME ${LIBPGMATH_LIBRARY_NAME})
endif()
install(TARGETS ${LIBPGMATH_LIBRARY_NAME}
  LIBRARY DESTINATION lib)
install(TARGETS ${LIBPGMATH_LIBRARY_NAME}_static
  ARCHIVE DESTINATION lib)
