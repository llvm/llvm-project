set( LLDB_USED_LIBS
  lldbBreakpoint
  lldbCommands
  lldbDataFormatters
  lldbHost
  lldbCore
  lldbExpression
  lldbInterpreter
  lldbSymbol
  lldbTarget
  lldbUtility

  # Plugins
  lldbPluginDisassemblerLLVM
  lldbPluginSymbolFileDWARF
  lldbPluginSymbolFileSymtab
  lldbPluginDynamicLoaderStatic
  lldbPluginDynamicLoaderPosixDYLD
  lldbPluginDynamicLoaderHexagonDYLD

  lldbPluginObjectFileMachO
  lldbPluginObjectFileELF
  lldbPluginObjectFileJIT
  lldbPluginSymbolVendorELF
  lldbPluginObjectContainerBSDArchive
  lldbPluginObjectContainerMachOArchive
  lldbPluginProcessGDBRemote
  lldbPluginProcessMachCore
  lldbPluginProcessUtility
  lldbPluginPlatformGDB
  lldbPluginPlatformFreeBSD
  lldbPluginPlatformKalimba
  lldbPluginPlatformLinux
  lldbPluginPlatformPOSIX
  lldbPluginPlatformWindows
  lldbPluginObjectFileMachO
  lldbPluginObjectContainerMachOArchive
  lldbPluginObjectContainerBSDArchive
  lldbPluginPlatformMacOSX
  lldbPluginDynamicLoaderMacOSXDYLD
  lldbPluginUnwindAssemblyInstEmulation
  lldbPluginUnwindAssemblyX86
  lldbPluginAppleObjCRuntime
  lldbPluginCXXItaniumABI
  lldbPluginABIMacOSX_arm
  lldbPluginABIMacOSX_arm64
  lldbPluginABIMacOSX_i386
  lldbPluginABISysV_x86_64
  lldbPluginABISysV_hexagon
  lldbPluginABISysV_ppc
  lldbPluginABISysV_ppc64
  lldbPluginInstructionARM
  lldbPluginInstructionARM64
  lldbPluginObjectFilePECOFF
  lldbPluginOSPython
  lldbPluginMemoryHistoryASan
  lldbPluginInstrumentationRuntimeAddressSanitizer
  )

# Need to export the API in the liblldb.dll for Windows
# The lldbAPI source files are added directly in liblldb
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows" )
  list(APPEND LLDB_USED_LIBS
    lldbAPI
    )
endif ()

# Windows-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  list(APPEND LLDB_USED_LIBS
    lldbPluginProcessWindows
    lldbPluginProcessElfCore
    lldbPluginJITLoaderGDB
    Ws2_32
    )
endif ()

# Linux-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
  list(APPEND LLDB_USED_LIBS
    lldbPluginProcessLinux
    lldbPluginProcessPOSIX
    lldbPluginProcessElfCore
    lldbPluginJITLoaderGDB
   )
endif ()

# FreeBSD-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
  list(APPEND LLDB_USED_LIBS
    lldbPluginProcessFreeBSD
    lldbPluginProcessPOSIX
    lldbPluginProcessElfCore
    lldbPluginJITLoaderGDB
    )
endif ()

# Darwin-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Darwin" )
  set(LLDB_VERS_GENERATED_FILE ${LLDB_BINARY_DIR}/source/LLDB_vers.c)
  add_custom_command(OUTPUT ${LLDB_VERS_GENERATED_FILE}
    COMMAND ${LLDB_SOURCE_DIR}/scripts/generate-vers.pl
            ${LLDB_SOURCE_DIR}/lldb.xcodeproj/project.pbxproj liblldb_core
            > ${LLDB_VERS_GENERATED_FILE})

  set_source_files_properties(${LLDB_VERS_GENERATED_FILE} PROPERTIES GENERATED 1)
  list(APPEND LLDB_USED_LIBS
    lldbPluginDynamicLoaderDarwinKernel
    lldbPluginProcessMacOSXKernel
    lldbPluginSymbolVendorMacOSX
    lldbPluginSystemRuntimeMacOSX
    lldbPluginProcessElfCore
    lldbPluginJITLoaderGDB
    )
endif()

set( CLANG_USED_LIBS
  clangAnalysis
  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangEdit
  clangFrontend
  clangLex
  clangParse
  clangRewrite
  clangRewriteFrontend
  clangSema
  clangSerialization
  )

set(LLDB_SYSTEM_LIBS)
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT __ANDROID_NDK__)
  if (NOT LLDB_DISABLE_LIBEDIT)
    list(APPEND LLDB_SYSTEM_LIBS edit)
  endif()
  if (NOT LLDB_DISABLE_CURSES)
    list(APPEND LLDB_SYSTEM_LIBS panel ncurses)
  endif()
endif()
# On FreeBSD backtrace() is provided by libexecinfo, not libc.
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  list(APPEND LLDB_SYSTEM_LIBS execinfo)
endif()

if (NOT LLDB_DISABLE_PYTHON AND NOT LLVM_BUILD_STATIC)
  list(APPEND LLDB_SYSTEM_LIBS ${PYTHON_LIBRARIES})
endif()

list(APPEND LLDB_SYSTEM_LIBS ${system_libs})

if (LLVM_BUILD_STATIC)
  list(APPEND LLDB_SYSTEM_LIBS python2.7 z util termcap gpm ssl crypto bsd)
endif()

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  interpreter
  asmparser
  bitreader
  bitwriter
  codegen
  ipo
  selectiondag
  bitreader
  mc
  mcjit
  core
  mcdisassembler
  executionengine
  option
  )

if ( NOT LLDB_DISABLE_PYTHON )
  set(LLDB_WRAP_PYTHON ${LLDB_BINARY_DIR}/scripts/LLDBWrapPython.cpp)

  set_source_files_properties(${LLDB_WRAP_PYTHON} PROPERTIES GENERATED 1)
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND
      NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    set_property(SOURCE ${LLDB_WRAP_PYTHON}
                 APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-sequence-point")
  endif ()
endif()
