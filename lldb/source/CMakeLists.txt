include_directories(.)

set(lldbBase_SOURCES
    lldb.cpp
  )

foreach(file
        "${LLDB_SOURCE_DIR}/.git/logs/HEAD" # Git
        "${LLDB_SOURCE_DIR}/.svn/wc.db"     # SVN 1.7
        "${LLDB_SOURCE_DIR}/.svn/entries"   # SVN 1.6
        )
  if(EXISTS "${file}")
    set(lldb_vc "${file}")
    break()
  endif()
endforeach()

set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/VCSRevision.h")
set(get_svn_script "${LLVM_CMAKE_PATH}/GenerateVersionFromVCS.cmake")

if(lldb_vc)
  set(lldb_source_dir ${LLDB_SOURCE_DIR})
endif()

add_custom_command(OUTPUT "${version_inc}"
  DEPENDS "${lldb_vc}" "${get_svn_script}"
  COMMAND ${CMAKE_COMMAND} "-DNAMES=LLDB"
                           "-DLLDB_SOURCE_DIR=${LLDB_SOURCE_DIR}"
                           "-DHEADER_FILE=${version_inc}"
                           -P "${get_svn_script}")

# Mark the generated header as being generated.
set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

list(APPEND lldbBase_SOURCES ${version_inc})

if(APPLE)
  set(apple_version_inc "${CMAKE_CURRENT_BINARY_DIR}/AppleVersion.inc")
  set(apple_version_script "${LLDB_SOURCE_DIR}/cmake/modules/EmbedAppleVersion.cmake")
  set(info_plist ${LLDB_SOURCE_DIR}/resources/LLDB-Info.plist)

  # Create custom target to generate the VC revision include.
  add_custom_command(OUTPUT "${apple_version_inc}"
    DEPENDS "${apple_version_script}" "${info_plist}"
    COMMAND
    ${CMAKE_COMMAND} "-DLLDB_INFO_PLIST=${info_plist}"
                     "-DHEADER_FILE=${apple_version_inc}"
                     -P "${apple_version_script}")

  # Mark the generated header as being generated.
  set_source_files_properties("${apple_version_inc}"
    PROPERTIES GENERATED TRUE
               HEADER_FILE_ONLY TRUE)

  # Tell Version.cpp that it needs to build with -DHAVE_SVN_VERSION_INC.
  set_property(SOURCE lldb.cpp APPEND PROPERTY 
               COMPILE_DEFINITIONS "HAVE_APPLE_VERSION_INC")
  list(APPEND lldbBase_SOURCES ${apple_version_inc})
elseif(LLDB_VERSION_STRING)
  set_property(SOURCE lldb.cpp APPEND PROPERTY
               COMPILE_DEFINITIONS "LLDB_VERSION_STRING=${LLDB_VERSION_STRING}")
endif()

add_lldb_library(lldbBase
  ${lldbBase_SOURCES}
  )

add_subdirectory(Breakpoint)
add_subdirectory(Commands)
add_subdirectory(Core)
add_subdirectory(DataFormatters)
add_subdirectory(Expression)
add_subdirectory(Host)
add_subdirectory(Initialization)
add_subdirectory(Interpreter)
add_subdirectory(Plugins)
add_subdirectory(Symbol)
add_subdirectory(Target)
add_subdirectory(Utility)

# Build API last.  Since liblldb needs to link against every other target, it needs
# those targets to have already been created.
add_subdirectory(API)
