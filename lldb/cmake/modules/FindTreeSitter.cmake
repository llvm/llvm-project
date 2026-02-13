# FindTreeSitter.cmake

include(FindPackageHandleStandardArgs)

find_path(TreeSitter_INCLUDE_DIR
  NAMES tree_sitter/api.h)

find_library(TreeSitter_LIBRARY
  NAMES tree-sitter treesitter)

find_package_handle_standard_args(TreeSitter
  REQUIRED_VARS TreeSitter_LIBRARY TreeSitter_INCLUDE_DIR
)

mark_as_advanced(
  TreeSitter_INCLUDE_DIR
  TreeSitter_LIBRARY
)
