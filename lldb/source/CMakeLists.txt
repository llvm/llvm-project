include_directories(.)

if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
include_directories(
  Plugins/Process/Linux
  Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
include_directories(
  Plugins/Process/FreeBSD
  Plugins/Process/POSIX
  )
endif ()

add_subdirectory(API)
add_subdirectory(Breakpoint)
add_subdirectory(Commands)
add_subdirectory(Core)
add_subdirectory(DataFormatters)
add_subdirectory(Expression)
add_subdirectory(Host)
add_subdirectory(Interpreter)
add_subdirectory(Plugins)
add_subdirectory(Symbol)
add_subdirectory(Target)
add_subdirectory(Utility)

set( LLDB_USED_LIBS
  lldbAPI
  lldbBreakpoint
  lldbCommands
  lldbDataFormatters
  lldbHostCommon
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
  
  lldbPluginObjectFileMachO
  lldbPluginObjectFileELF
  lldbPluginObjectContainerBSDArchive
  lldbPluginObjectContainerMachOArchive
  lldbPluginProcessGDBRemote  
  lldbPluginProcessMachCore
  lldbPluginProcessUtility
  lldbPluginPlatformGDB
  lldbPluginPlatformFreeBSD
  lldbPluginPlatformLinux
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
  lldbPluginABIMacOSX_i386
  lldbPluginABISysV_x86_64
  lldbPluginInstructionARM
  lldbPluginObjectFilePECOFF
  lldbPluginOSPython
  )

# Windows-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  list(APPEND LLDB_USED_LIBS
    #lldbHostWindows
    #lldbPluginPlatformWindows
    #Ws2_32
    )
endif ()

# Linux-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
  list(APPEND LLDB_USED_LIBS
    lldbHostLinux
    lldbPluginProcessLinux
    lldbPluginProcessPOSIX
    )
endif ()

# FreeBSD-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
  list(APPEND LLDB_USED_LIBS
    lldbHostFreeBSD
    lldbPluginProcessFreeBSD
    lldbPluginProcessPOSIX
    )
endif ()

# Darwin-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Darwin" )
  set(LLDB_VERS_GENERATED_FILE ${LLDB_BINARY_DIR}/source/LLDB_vers.c)
  add_custom_command(OUTPUT ${LLDB_VERS_GENERATED_FILE}
    COMMAND ${LLDB_SOURCE_DIR}/scripts/generate-vers.pl
            ${LLDB_SOURCE_DIR}/lldb.xcodeproj/project.pbxproj
            > ${LLDB_VERS_GENERATED_FILE})

  set_source_files_properties(${LLDB_VERS_GENERATED_FILE} PROPERTIES GENERATED 1)
  list(APPEND LLDB_USED_LIBS
    lldbHostMacOSX
    lldbPluginDynamicLoaderDarwinKernel
    lldbPluginProcessMacOSXKernel
    lldbPluginSymbolVendorMacOSX
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
  clangRewriteCore
  clangRewriteFrontend
  clangSema
  clangSerialization
  )  
  
set( LLDB_SYSTEM_LIBS
  edit
  python2.7
  )

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  jit
  interpreter
  nativecodegen
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
  )

set_source_files_properties(${LLDB_BINARY_DIR}/scripts/LLDBWrapPython.cpp PROPERTIES GENERATED 1)
set(SHARED_LIBRARY 1)

add_lldb_library(liblldb
  lldb.cpp
  lldb-log.cpp
  ${LLDB_BINARY_DIR}/scripts/LLDBWrapPython.cpp
  ${LLDB_VERS_GENERATED_FILE}
  )
set_target_properties(liblldb
  PROPERTIES
  OUTPUT_NAME lldb
  VERSION ${LLDB_VERSION}
  )
add_dependencies(liblldb
  ${LLDB_BINARY_DIR}/scripts/LLDBWrapPython.cpp
  ${LLDB_VERS_GENERATED_FILE}
  )
target_link_libraries(liblldb ${LLDB_SYSTEM_LIBS})

# Determine LLDB revision and repository. GetSourceVersion and GetRepositoryPath are shell-scripts, and as
# such will not work on Windows.
if ( NOT CMAKE_SYSTEM_NAME MATCHES "Windows" )
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/utils/GetSourceVersion ${LLDB_SOURCE_DIR}
                  OUTPUT_VARIABLE LLDB_REVISION)
  if ( LLDB_REVISION )
    string(REGEX REPLACE "(\r?\n)+$" "" LLDB_REVISION ${LLDB_REVISION})
  endif()

  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/utils/GetRepositoryPath ${LLDB_SOURCE_DIR}
                  OUTPUT_VARIABLE LLDB_REPOSITORY)
  if ( LLDB_REPOSITORY )
    # Replace newline characters with spaces
    string(REGEX REPLACE "(\r?\n)+" " " LLDB_REPOSITORY ${LLDB_REPOSITORY})

    # Remove trailing spaces
    string(REGEX REPLACE "(\ )+$" "" LLDB_REPOSITORY ${LLDB_REPOSITORY})
  endif()

  set_property(
    SOURCE lldb.cpp
    PROPERTY COMPILE_DEFINITIONS "LLDB_REVISION=\"${LLDB_REVISION}\"" "LLDB_REPOSITORY=\"${LLDB_REPOSITORY}\"")
endif ()
# FIXME: implement svn/git revision and repository parsing solution on Windows. There is an SVN-only
#        revision parsing solution in tools/clang/lib/Basic/CMakelists.txt.


install(TARGETS liblldb
  LIBRARY DESTINATION lib)
