set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(diagtool
  diagtool_main.cpp
  DiagTool.cpp
  DiagnosticNames.cpp
  FindDiagnosticID.cpp
  ListWarnings.cpp
  ShowEnabledWarnings.cpp
  TreeView.cpp
)

target_link_libraries(diagtool
  PRIVATE
  clangBasic
  clangFrontend
  )

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(TARGETS diagtool
    COMPONENT diagtool
    RUNTIME DESTINATION bin)

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_llvm_install_targets(install-diagtool
      DEPENDS diagtool
      COMPONENT diagtool)
  endif()
endif()
