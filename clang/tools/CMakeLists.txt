add_subdirectory(libclang)
add_subdirectory(c-index-test)
add_subdirectory(arcmt-test)
add_subdirectory(c-arcmt-test)
add_subdirectory(diagtool)
add_subdirectory(driver)
add_subdirectory(clang-check)

# We support checking out the clang-tools-extra repository into the 'extra'
# subdirectory. It contains tools developed as part of the Clang/LLVM project
# on top of the Clang tooling platform. We keep them in a separate repository
# to keep the primary Clang repository small and focused.
# It also may be included by LLVM_EXTERNAL_CLANG_TOOLS_EXTRA_SOURCE_DIR.
add_llvm_external_project(clang-tools-extra extra)
