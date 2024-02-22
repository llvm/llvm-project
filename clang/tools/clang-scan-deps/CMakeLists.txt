set(LLVM_LINK_COMPONENTS
  Core
  Option
  Support
  CAS
  TargetParser
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ScanDepsOptsTableGen)

add_clang_tool(clang-scan-deps
  ClangScanDeps.cpp

  DEPENDS
  ScanDepsOptsTableGen
  GENERATE_DRIVER
  )

set(CLANG_SCAN_DEPS_LIB_DEPS
  clangAST
  clangBasic
  clangCAS
  clangDependencyScanning
  clangDriver
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  )

clang_target_link_libraries(clang-scan-deps
  PRIVATE
  ${CLANG_SCAN_DEPS_LIB_DEPS}
  )

add_custom_command(OUTPUT ${LLVM_TOOLS_BINARY_DIR}/clang-deps-launcher.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/clang-deps-launcher.py
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/clang-deps-launcher.py"
          "${LLVM_TOOLS_BINARY_DIR}/clang-deps-launcher.py"
  COMMENT "Copy clang-deps-launcher.py..."
  )

add_custom_target(clang-deps-launcher ALL DEPENDS ${LLVM_TOOLS_BINARY_DIR}/clang-deps-launcher.py)
install(
  FILES ${LLVM_TOOLS_BINARY_DIR}/clang-deps-launcher.py
  DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  COMPONENT clang-deps-launcher)
add_llvm_install_targets(install-clang-deps-launcher
                         DEPENDS clang-deps-launcher
                         COMPONENT clang-deps-launcher)
