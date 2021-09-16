include_directories(.)

set(lldbBase_SOURCES
    lldb.cpp
  )


find_first_existing_vc_file("${LLDB_SOURCE_DIR}" lldb_vc)

set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/VCSVersion.inc")
set(generate_vcs_version_script "${LLVM_CMAKE_DIR}/GenerateVersionFromVCS.cmake")

if(lldb_vc AND LLVM_APPEND_VC_REV)
  set(lldb_source_dir ${LLDB_SOURCE_DIR})
endif()

add_custom_command(OUTPUT "${version_inc}"
  DEPENDS "${lldb_vc}" "${generate_vcs_version_script}"
  COMMAND ${CMAKE_COMMAND} "-DNAMES=LLDB"
                           "-DLLDB_SOURCE_DIR=${lldb_source_dir}"
                           "-DHEADER_FILE=${version_inc}"
                           -P "${generate_vcs_version_script}")

# Mark the generated header as being generated.
set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

list(APPEND lldbBase_SOURCES ${version_inc})

if(LLDB_VERSION_STRING)
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
