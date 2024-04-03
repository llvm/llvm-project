# Creates the ClangdXPC framework.
macro(create_clangd_xpc_framework target name)
  set(CLANGD_FRAMEWORK_LOCATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${name}.framework")
  set(CLANGD_FRAMEWORK_OUT_LOCATION "${CLANGD_FRAMEWORK_LOCATION}/Versions/A")

  # Create the framework info PLIST.
  set(CLANGD_XPC_FRAMEWORK_NAME "${name}")
  configure_file(
    "${CLANGD_XPC_SOURCE_DIR}/cmake/Info.plist.in"
    "${CLANGD_XPC_BINARY_DIR}/${name}.Info.plist")

  set(CLANGD_XPC_SERVICE_NAME "clangd")
  set(CLANGD_XPC_SERVICE_OUT_LOCATION
      "${CLANGD_FRAMEWORK_OUT_LOCATION}/XPCServices/${CLANGD_XPC_SERVICE_NAME}.xpc/Contents")

  # Create the XPC service info PLIST.
  set(CLANGD_XPC_SERVICE_BUNDLE_NAME "org.llvm.${CLANGD_XPC_SERVICE_NAME}")
  configure_file(
    "${CLANGD_XPC_SOURCE_DIR}/cmake/XPCServiceInfo.plist.in"
    "${CLANGD_XPC_BINARY_DIR}/${name}Service.Info.plist")

  # Create the custom command
  add_custom_command(OUTPUT ${CLANGD_FRAMEWORK_LOCATION}
    # Copy the PLIST.
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CLANGD_XPC_BINARY_DIR}/${name}.Info.plist"
      "${CLANGD_FRAMEWORK_OUT_LOCATION}/Resources/Info.plist"

    # Copy the framework binary.
    COMMAND ${CMAKE_COMMAND} -E copy
       "$<TARGET_FILE:${target}>"
       "${CLANGD_FRAMEWORK_OUT_LOCATION}/${name}"

    # Copy the XPC Service PLIST.
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CLANGD_XPC_BINARY_DIR}/${name}Service.Info.plist"
      "${CLANGD_XPC_SERVICE_OUT_LOCATION}/Info.plist"

    # Copy the Clangd binary.
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:clangd>"
      "${CLANGD_XPC_SERVICE_OUT_LOCATION}/MacOS/clangd"

     COMMAND ${CMAKE_COMMAND} -E create_symlink "A"
     "${CLANGD_FRAMEWORK_LOCATION}/Versions/Current"

     COMMAND ${CMAKE_COMMAND} -E create_symlink
     "Versions/Current/Resources"
     "${CLANGD_FRAMEWORK_LOCATION}/Resources"

     COMMAND ${CMAKE_COMMAND} -E create_symlink
     "Versions/Current/XPCServices"
     "${CLANGD_FRAMEWORK_LOCATION}/XPCServices"

     COMMAND ${CMAKE_COMMAND} -E create_symlink
     "Versions/Current/${name}"
     "${CLANGD_FRAMEWORK_LOCATION}/${name}"

    DEPENDS
      "${CLANGD_XPC_BINARY_DIR}/${name}.Info.plist"
      "${CLANGD_XPC_BINARY_DIR}/${name}Service.Info.plist"
      clangd
    COMMENT "Creating ClangdXPC framework"
    VERBATIM
  )

  add_custom_target(
    ClangdXPC
    DEPENDS
    ${target}
    ${CLANGD_FRAMEWORK_LOCATION}
  )

  set_property(
    TARGET ClangdXPC
    APPEND
    PROPERTY ADDITIONAL_CLEAN_FILES ${CLANGD_FRAMEWORK_LOCATION}
  )

  # clangd is already signed as a standalone executable, so it must be forced.
  llvm_codesign(ClangdXPC BUNDLE_PATH "${CLANGD_FRAMEWORK_OUT_LOCATION}/XPCServices/${CLANGD_XPC_SERVICE_NAME}.xpc/" FORCE)
  # ClangdXPC library is already signed as a standalone library, so it must be forced.
  llvm_codesign(ClangdXPC BUNDLE_PATH "${CLANGD_FRAMEWORK_LOCATION}" FORCE)
endmacro(create_clangd_xpc_framework)
