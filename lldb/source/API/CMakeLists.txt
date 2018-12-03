if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  add_definitions( -DEXPORT_LIBLLDB )
endif()

get_property(LLDB_ALL_PLUGINS GLOBAL PROPERTY LLDB_PLUGINS)

add_lldb_library(liblldb SHARED
  SBAddress.cpp
  SBAttachInfo.cpp
  SBBlock.cpp
  SBBreakpoint.cpp
  SBBreakpointLocation.cpp
  SBBreakpointName.cpp
  SBBreakpointOptionCommon.cpp
  SBBroadcaster.cpp
  SBCommandInterpreter.cpp
  SBCommandReturnObject.cpp
  SBCommunication.cpp
  SBCompileUnit.cpp
  SBData.cpp
  SBDebugger.cpp
  SBDeclaration.cpp
  SBError.cpp
  SBEvent.cpp
  SBExecutionContext.cpp
  SBExpressionOptions.cpp
  SBFileSpec.cpp
  SBFileSpecList.cpp
  SBFrame.cpp
  SBFunction.cpp
  SBHostOS.cpp
  SBInitializerOptions.cpp
  SBInstruction.cpp
  SBInstructionList.cpp
  SBLanguageRuntime.cpp
  SBLaunchInfo.cpp
  SBLineEntry.cpp
  SBListener.cpp
  SBMemoryRegionInfo.cpp
  SBMemoryRegionInfoList.cpp
  SBModule.cpp
  SBModuleSpec.cpp
  SBPlatform.cpp
  SBProcess.cpp
  SBProcessInfo.cpp
  SBQueue.cpp
  SBQueueItem.cpp
  SBSection.cpp
  SBSourceManager.cpp
  SBStream.cpp
  SBStringList.cpp
  SBStructuredData.cpp
  SBSymbol.cpp
  SBSymbolContext.cpp
  SBSymbolContextList.cpp
  SBTarget.cpp
  SBThread.cpp
  SBThreadCollection.cpp
  SBThreadPlan.cpp
  SBTrace.cpp
  SBTraceOptions.cpp
  SBType.cpp
  SBTypeCategory.cpp
  SBTypeEnumMember.cpp
  SBTypeFilter.cpp
  SBTypeFormat.cpp
  SBTypeNameSpecifier.cpp
  SBTypeSummary.cpp
  SBTypeSynthetic.cpp
  SBValue.cpp
  SBValueList.cpp
  SBVariablesOptions.cpp
  SBWatchpoint.cpp
  SBUnixSignals.cpp
  SystemInitializerFull.cpp
  ${LLDB_WRAP_PYTHON}

  LINK_LIBS
    lldbBase
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInitialization
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    ${LLDB_ALL_PLUGINS}
  LINK_COMPONENTS
    Support
  )

add_dependencies(lldb-suite liblldb)

if (MSVC)
  set_property(SOURCE ${LLDB_WRAP_PYTHON} APPEND_STRING PROPERTY COMPILE_FLAGS " /W0")
else()
  set_property(SOURCE ${LLDB_WRAP_PYTHON} APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
endif()

set_source_files_properties(${LLDB_WRAP_PYTHON} PROPERTIES GENERATED 1)
if (CLANG_CL)
  set_property(SOURCE ${LLDB_WRAP_PYTHON} APPEND_STRING
    PROPERTY COMPILE_FLAGS " -Wno-unused-function")
endif()
if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND
    NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
  set_property(SOURCE ${LLDB_WRAP_PYTHON} APPEND_STRING
    PROPERTY COMPILE_FLAGS " -Wno-sequence-point -Wno-cast-qual")
endif ()

set_target_properties(liblldb
  PROPERTIES
  VERSION ${LLDB_VERSION}
)

if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  if (NOT LLDB_EXPORT_ALL_SYMBOLS)
    # If we're not exporting all symbols, we'll want to explicitly set
    # the exported symbols here.  This prevents 'log enable --stack ...'
    # from working on some systems but limits the liblldb size.
    MESSAGE("-- Symbols (liblldb): exporting all symbols from the lldb namespace")
    add_llvm_symbol_exports(liblldb ${CMAKE_CURRENT_SOURCE_DIR}/liblldb.exports)
  else()
    # Don't use an explicit export.  Instead, tell the linker to
    # export all symbols.
    MESSAGE("-- Symbols (liblldb): exporting all symbols from the lldb and lldb_private namespaces")
    add_llvm_symbol_exports(liblldb ${CMAKE_CURRENT_SOURCE_DIR}/liblldb-private.exports)
  endif()
endif()

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  # Only MSVC has the ABI compatibility problem and avoids using FindPythonLibs,
  # so only it needs to explicitly link against ${PYTHON_LIBRARY}
  if (MSVC AND NOT LLDB_DISABLE_PYTHON)
    target_link_libraries(liblldb PRIVATE ${PYTHON_LIBRARY})
  endif()
else()
  set_target_properties(liblldb
    PROPERTIES
    OUTPUT_NAME lldb
  )
endif()

if (LLDB_BUILD_FRAMEWORK)
  set_target_properties(liblldb
    PROPERTIES
    SOVERSION "1.0.0"
    OUTPUT_NAME LLDB
    FRAMEWORK On
    FRAMEWORK_VERSION ${LLDB_FRAMEWORK_VERSION}
    MACOSX_FRAMEWORK_INFO_PLIST ${LLDB_SOURCE_DIR}/resources/LLDB-Info.plist
    LIBRARY_OUTPUT_DIRECTORY ${LLDB_FRAMEWORK_DIR}
  )
endif()

if (LLDB_WRAP_PYTHON)
  add_dependencies(liblldb swig_wrapper)
endif()
