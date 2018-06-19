file(GLOB public_headers ${LLDB_SOURCE_DIR}/include/lldb/API/*.h)
file(GLOB root_public_headers ${LLDB_SOURCE_DIR}/include/lldb/lldb-*.h)
file(GLOB root_private_headers ${LLDB_SOURCE_DIR}/include/lldb/lldb-private*.h)
list(REMOVE_ITEM root_public_headers ${root_private_headers})
foreach(header
    ${public_headers}
    ${root_public_headers}
    ${LLDB_SOURCE_DIR}/include/lldb/Utility/SharingPtr.h)
  get_filename_component(basename ${header} NAME)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders/${basename}
                     DEPENDS ${header}
                     COMMAND ${CMAKE_COMMAND} -E copy ${header} ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders/${basename})
  list(APPEND framework_headers ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders/${basename})
endforeach()
add_custom_target(lldb-framework-headers
  DEPENDS ${framework_headers}
  COMMAND ${LLDB_SOURCE_DIR}/scripts/framework-header-fix.sh ${CMAKE_CURRENT_BINARY_DIR} ${LLDB_VERSION})

if (NOT IOS)
  if (NOT LLDB_BUILT_STANDALONE)
    add_dependencies(lldb-framework clang-headers)
  endif()
  add_custom_command(TARGET lldb-framework POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders $<TARGET_FILE_DIR:liblldb>/Headers
    COMMAND ${CMAKE_COMMAND} -E create_symlink Versions/Current/Headers ${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Headers
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${LLDB_FRAMEWORK_VERSION} ${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Versions/Current
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/clang/${LLDB_VERSION} $<TARGET_FILE_DIR:liblldb>/Resources/Clang
  )
else()
  add_custom_command(TARGET lldb-framework POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders $<TARGET_FILE_DIR:liblldb>/Headers
  )
endif()

set_target_properties(liblldb PROPERTIES
  OUTPUT_NAME LLDB
  FRAMEWORK On
  FRAMEWORK_VERSION ${LLDB_FRAMEWORK_VERSION}
  MACOSX_FRAMEWORK_INFO_PLIST ${LLDB_SOURCE_DIR}/resources/LLDB-Info.plist
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}
  PUBLIC_HEADER "${framework_headers}")

add_dependencies(lldb-framework
  liblldb
  lldb-framework-headers)
