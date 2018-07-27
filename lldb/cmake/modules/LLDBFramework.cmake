# We intentionally do not use CMake's framework support because of a bug in
# CMake. See: https://gitlab.kitware.com/cmake/cmake/issues/18216

# We set up part of the framework structure first, as some scripts and build
# rules assume they exist already.
file(MAKE_DIRECTORY ${LLDB_FRAMEWORK_DIR} ${LLDB_FRAMEWORK_DIR}/${LLDB_FRAMEWORK_RESOURCE_DIR})
execute_process(
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink ${LLDB_FRAMEWORK_VERSION} ${LLDB_FRAMEWORK_DIR}/Versions/Current
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink Versions/Current/Resources ${LLDB_FRAMEWORK_DIR}/Resources
)

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

add_custom_target(lldb-framework-headers DEPENDS ${framework_headers})

add_custom_command(TARGET lldb-framework POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/FrameworkHeaders ${LLDB_FRAMEWORK_DIR}/${LLDB_FRAMEWORK_HEADER_DIR}
  COMMAND ${LLDB_SOURCE_DIR}/scripts/framework-header-fix.sh ${LLDB_FRAMEWORK_DIR}/${LLDB_FRAMEWORK_HEADER_DIR} ${LLDB_VERSION}
)

if (NOT IOS)
  if (NOT LLDB_BUILT_STANDALONE)
    add_dependencies(lldb-framework clang-headers)
  endif()
  add_custom_command(TARGET lldb-framework POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink Versions/Current/Headers ${LLDB_FRAMEWORK_DIR}/Headers
    COMMAND ${CMAKE_COMMAND} -E create_symlink Versions/Current/LLDB ${LLDB_FRAMEWORK_DIR}/LLDB
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${LLDB_FRAMEWORK_VERSION} ${LLDB_FRAMEWORK_DIR}/Versions/Current
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/clang/${LLDB_VERSION} ${LLDB_FRAMEWORK_DIR}/${LLDB_FRAMEWORK_RESOURCE_DIR}/Clang
  )
endif()

# These are used to fill out LLDB-Info.plist. These are relevant when building
# the framework, and must be defined before configuring Info.plist.
set(PRODUCT_NAME "LLDB")
set(EXECUTABLE_NAME "LLDB")
set(CURRENT_PROJECT_VERSION "360.99.0")
configure_file(${LLDB_SOURCE_DIR}/resources/LLDB-Info.plist
  ${LLDB_FRAMEWORK_DIR}/${LLDB_FRAMEWORK_RESOURCE_DIR}/Info.plist)

add_dependencies(lldb-framework
  lldb-framework-headers
  lldb-suite)
