# Specialize the clang-doc target to avoid building other projects
add_lit_testsuite(check-clang-extra-clang-doc "Running clang-doc tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS clang-doc
  DEPENDS ${LLVM_UTILS_DEPS}
)
