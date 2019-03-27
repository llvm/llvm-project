# Path relative to the root binary directory
get_filename_component(
  framework_target_dir ${LLDB_FRAMEWORK_BUILD_DIR} ABSOLUTE
  BASE_DIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

message(STATUS "LLDB.framework: build path is '${framework_target_dir}'")
message(STATUS "LLDB.framework: install path is '${LLDB_FRAMEWORK_INSTALL_DIR}'")
message(STATUS "LLDB.framework: resources subdirectory is 'Versions/${LLDB_FRAMEWORK_VERSION}/Resources'")

# Configure liblldb as a framework bundle
set_target_properties(liblldb PROPERTIES
  FRAMEWORK ON
  FRAMEWORK_VERSION ${LLDB_FRAMEWORK_VERSION}

  OUTPUT_NAME LLDB
  VERSION ${LLDB_VERSION}
  LIBRARY_OUTPUT_DIRECTORY ${framework_target_dir}

  # Compatibility version
  SOVERSION "1.0.0"

  MACOSX_FRAMEWORK_IDENTIFIER com.apple.LLDB.framework
  MACOSX_FRAMEWORK_BUNDLE_VERSION ${LLDB_VERSION}
  MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${LLDB_VERSION}
  MACOSX_FRAMEWORK_INFO_PLIST ${LLDB_SOURCE_DIR}/resources/LLDB-Info.plist.in
)

# Affects the layout of the framework bundle (default is macOS layout).
if(IOS)
  set_target_properties(liblldb PROPERTIES
    XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "${IPHONEOS_DEPLOYMENT_TARGET}")
else()
  set_target_properties(liblldb PROPERTIES
    XCODE_ATTRIBUTE_MACOSX_DEPLOYMENT_TARGET "${MACOSX_DEPLOYMENT_TARGET}")
endif()

# Target to capture extra steps for a fully functional framework bundle.
add_custom_target(lldb-framework)
add_dependencies(lldb-framework liblldb)

# Dependencies are defined once tools are added (see AddLLDB.cmake)
if(LLDB_FRAMEWORK_TOOLS)
  message(STATUS "LLDB.framework: adding tools ${LLDB_FRAMEWORK_TOOLS}")
  foreach(tool ${LLDB_FRAMEWORK_TOOLS})
    add_custom_command(TARGET lldb-framework POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${tool}> $<TARGET_FILE_DIR:liblldb>/Resources
      COMMENT "LLDB.framework: copy additional tool ${tool}"
    )
  endforeach()
else()
  message(WARNING "LLDB.framework: no additional tools configured (set via LLDB_FRAMEWORK_TOOLS)")
endif()

# Apart from this one, CMake creates all required symlinks in the framework bundle.
add_custom_command(TARGET lldb-framework POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          Versions/Current/Headers
          ${framework_target_dir}/LLDB.framework/Headers
  COMMENT "LLDB.framework: create Headers symlink"
)

# At configuration time, collect headers for the framework bundle and copy them
# into a staging directory. Later we can copy over the entire folder.
file(GLOB public_headers ${LLDB_SOURCE_DIR}/include/lldb/API/*.h)
file(GLOB root_public_headers ${LLDB_SOURCE_DIR}/include/lldb/lldb-*.h)
file(GLOB root_private_headers ${LLDB_SOURCE_DIR}/include/lldb/lldb-private*.h)
list(REMOVE_ITEM root_public_headers ${root_private_headers})

set(lldb_header_staging ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders)
foreach(header
    ${public_headers}
    ${root_public_headers}
    ${LLDB_SOURCE_DIR}/include/lldb/Utility/SharingPtr.h)

  get_filename_component(basename ${header} NAME)
  set(staged_header ${lldb_header_staging}/${basename})

  add_custom_command(
    DEPENDS ${header} OUTPUT ${staged_header}
    COMMAND ${CMAKE_COMMAND} -E copy ${header} ${staged_header}
    COMMENT "LLDB.framework: collect framework header")

  list(APPEND lldb_staged_headers ${staged_header})
endforeach()

# Wrap output in a target, so lldb-framework can depend on it.
add_custom_target(lldb-framework-headers DEPENDS ${lldb_staged_headers})
add_dependencies(lldb-framework lldb-framework-headers)

# At build time, copy the staged headers into the framework bundle (and do
# some post-processing in-place).
add_custom_command(TARGET lldb-framework-headers POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${lldb_header_staging} $<TARGET_FILE_DIR:liblldb>/Headers
  COMMAND ${LLDB_SOURCE_DIR}/scripts/framework-header-fix.sh $<TARGET_FILE_DIR:liblldb>/Headers ${LLDB_VERSION}
  COMMENT "LLDB.framework: copy framework headers"
)

# Copy vendor-specific headers from clang (without staging).
if(NOT IOS AND NOT LLDB_BUILT_STANDALONE)
  add_dependencies(lldb-framework clang-resource-headers)
  add_custom_command(TARGET lldb-framework POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            $<TARGET_PROPERTY:clang-resource-headers,RUNTIME_OUTPUT_DIRECTORY>
            $<TARGET_FILE_DIR:liblldb>/Resources/Clang/include
    COMMENT "LLDB.framework: copy clang vendor-specific headers"
  )
endif()
