set(LLVM_NO_RTTI 1)

include(AddLLVM)
include(SwiftAddCustomCommandTarget)

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  add_definitions( -DEXPORT_LIBLLDB )
endif()

get_property(LLDB_ALL_PLUGINS GLOBAL PROPERTY LLDB_PLUGINS)

if(NOT LLDB_DISABLE_PYTHON)
  get_target_property(lldb_scripts_dir swig_wrapper BINARY_DIR)
  set(lldb_python_wrapper ${lldb_scripts_dir}/LLDBWrapPython.cpp)
endif()

if(LLDB_BUILD_FRAMEWORK AND LLVM_EXTERNALIZE_DEBUGINFO)
  set(LLVM_EXTERNALIZE_DEBUGINFO_EXTENSION framework.dSYM)
endif()

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
  SBReproducer.cpp
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
  ${lldb_python_wrapper}

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
    ${SWIFT_ALL_LIBS}
  LINK_COMPONENTS
    Support
  )

if (MSVC)
  set_source_files_properties(SBReproducer.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

if(lldb_python_wrapper)
  add_dependencies(liblldb swig_wrapper)

  if (MSVC)
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING PROPERTY COMPILE_FLAGS " /W0")
  else()
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
  endif()

  set_source_files_properties(${lldb_python_wrapper} PROPERTIES GENERATED ON)
  if (CLANG_CL)
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING
      PROPERTY COMPILE_FLAGS " -Wno-unused-function")
  endif()
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND
      NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING
      PROPERTY COMPILE_FLAGS " -Wno-sequence-point -Wno-cast-qual")
  endif ()
endif()

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
  set_target_properties(liblldb_exports PROPERTIES FOLDER "lldb misc")
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

if(LLDB_BUILD_FRAMEWORK)
  include(LLDBFramework)
endif()

set(lib_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX}")

if(EXISTS ${LLDB_PATH_TO_SWIFT_BUILD}/lib${LLVM_LIBDIR_SUFFIX}/swift/clang)
  set(CLANG_RESOURCE_PATH "${LLDB_PATH_TO_SWIFT_BUILD}/lib${LLVM_LIBDIR_SUFFIX}/swift/clang")
elseif(LLDB_BUILT_STANDALONE)
  set(CLANG_VERSION ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH})
  set(CLANG_RESOURCE_PATH "${LLDB_PATH_TO_CLANG_BUILD}/lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}")
  if(NOT EXISTS ${CLANG_RESOURCE_PATH})
    message(SEND_ERROR "standalone build cannot locate clang resource directory")
  endif()
elseif(LLVM_LIBRARY_OUTPUT_INTDIR)
  set(CLANG_RESOURCE_PATH "${LLVM_LIBRARY_OUTPUT_INTDIR}/swift/clang")
else()
  message(SEND_ERROR "unified build cannot locate clang resource directory")
endif()

set(clang_headers_target ${CLANG_RESOURCE_PATH}/include)
set(SWIFT_RESOURCE_PATH "${LLDB_PATH_TO_SWIFT_BUILD}/lib${LLVM_LIBDIR_SUFFIX}/swift")
if(NOT LLDB_BUILT_STANDALONE)
  set(clang_headers_target clang-headers symlink_clang_headers)
endif()

# Copy the clang resource directory.
add_custom_command_target(
    copy-clang-resource-dir
    COMMAND "${CMAKE_COMMAND}" "-E" "copy_directory" "${CLANG_RESOURCE_PATH}" "${lib_dir}/lldb/clang/"
    OUTPUT "${lib_dir}/lldb/clang/"
    VERBATIM
    ALL
    DEPENDS ${clang_headers_target})

# Only copy the swift resource directory if you are building lldb standalone.
# Otherwise, just rely on the swift libdir available from the swift build.
if(LLDB_BUILT_STANDALONE)
  add_custom_command_target(
      copy-swift-resource-dir
      COMMAND "${CMAKE_COMMAND}" "-E" "copy_directory" "${SWIFT_RESOURCE_PATH}" "${lib_dir}/lldb/swift/"
      OUTPUT "${lib_dir}/lldb/swift/"
      VERBATIM
      ALL
      DEPENDS ${SWIFT_RESOURCE_PATH})
endif()

add_dependencies(liblldb ${copy-clang-resource-dir} ${copy-swift-resource-dir})

install(
  CODE "file(MAKE_DIRECTORY ${lib_dir}/lldb)")

install(
  DIRECTORY "${lib_dir}/lldb/clang"
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/lldb/)

if(LLDB_BUILT_STANDALONE)
  install(
    DIRECTORY "${lib_dir}/lldb/swift"
    DESTINATION lib${LLVM_LIBDIR_SUFFIX}/lldb/)
endif()

