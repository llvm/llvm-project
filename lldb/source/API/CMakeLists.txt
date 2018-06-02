if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  add_definitions( -DEXPORT_LIBLLDB )
endif()

option(LLDB_BUILD_FRAMEWORK "Build the Darwin LLDB.framework" Off)

if(LLDB_BUILD_FRAMEWORK AND CMAKE_VERSION VERSION_LESS 3.7)
  message(FATAL_ERROR "LLDB_BUILD_FRAMEWORK is not supported on CMake < 3.7")
endif()

if (LLDB_BUILD_FRAMEWORK AND NOT APPLE)
  message(FATAL_ERROR "LLDB.framework cannot be generated unless targeting Apple platforms.")
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

if (LLVM_ENABLE_WERROR)
  if (MSVC)
    set_property(SOURCE ${LLDB_WRAP_PYTHON} APPEND_STRING PROPERTY COMPILE_FLAGS " /W0")
  else()
    set_property(SOURCE ${LLDB_WRAP_PYTHON} APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
  endif()
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

target_link_libraries(liblldb PRIVATE
  lldbPluginScriptInterpreterNone
  lldbPluginScriptInterpreterPython
  )

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

if (LLDB_WRAP_PYTHON)
  add_dependencies(liblldb swig_wrapper)
endif()

if(LLDB_BUILD_FRAMEWORK)
  file(GLOB public_headers ${LLDB_SOURCE_DIR}/include/lldb/API/*.h)
  file(GLOB root_public_headers ${LLDB_SOURCE_DIR}/include/lldb/lldb-*.h)
  file(GLOB root_private_headers ${LLDB_SOURCE_DIR}/include/lldb/lldb-private*.h)
  list(REMOVE_ITEM root_public_headers ${root_private_headers})

  foreach(header ${public_headers} ${root_public_headers} ${LLDB_SOURCE_DIR}/include/lldb/Utility/SharingPtr.h)
    get_filename_component(basename ${header} NAME)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders/${basename}
                       DEPENDS ${header}
                       COMMAND ${CMAKE_COMMAND} -E copy ${header} ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders/${basename})
    list(APPEND framework_headers ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders/${basename})
  endforeach()

  add_custom_target(lldb-framework-headers DEPENDS ${framework_headers}
                    COMMAND ${LLDB_SOURCE_DIR}/scripts/framework-header-fix.sh ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders ${LLDB_VERSION})
  add_dependencies(liblldb lldb-framework-headers)

  set_target_properties(liblldb PROPERTIES
    OUTPUT_NAME LLDB
    FRAMEWORK On
    FRAMEWORK_VERSION ${LLDB_FRAMEWORK_VERSION}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}
    PUBLIC_HEADER "${framework_headers}")

  if(NOT IOS)
    if (NOT LLDB_BUILT_STANDALONE)
      add_dependencies(liblldb clang-headers)
    endif()
    add_custom_command(TARGET liblldb POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders $<TARGET_FILE_DIR:liblldb>/Headers
      COMMAND ${CMAKE_COMMAND} -E create_symlink Versions/Current/Headers ${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Headers
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${LLDB_FRAMEWORK_VERSION} ${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Versions/Current
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/clang/${LLDB_VERSION} $<TARGET_FILE_DIR:liblldb>/Resources/Clang
      )
  else()
    add_custom_command(TARGET liblldb POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders $<TARGET_FILE_DIR:liblldb>/Headers
      )
  endif()

endif()

