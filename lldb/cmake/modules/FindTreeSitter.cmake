# FindTreeSitter.cmake

include(FindPackageHandleStandardArgs)

find_path(TreeSitter_INCLUDE_DIR
  NAMES tree_sitter/api.h)

find_library(TreeSitter_LIBRARY
  NAMES tree-sitter treesitter)

find_program(TreeSitter_BINARY
  NAMES tree-sitter)

if(TreeSitter_BINARY)
  execute_process(
    COMMAND "${TreeSitter_BINARY}" --version
    OUTPUT_VARIABLE TreeSitter_VERSION_OUTPUT
    ERROR_QUIET
    RESULT_VARIABLE TreeSitter_VERSION_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(TreeSitter_VERSION_RESULT EQUAL 0)
    string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+"
      TreeSitter_VERSION "${TreeSitter_VERSION_OUTPUT}")
  endif()
endif()

find_package_handle_standard_args(TreeSitter
  REQUIRED_VARS TreeSitter_LIBRARY TreeSitter_INCLUDE_DIR TreeSitter_BINARY
  VERSION_VAR TreeSitter_VERSION
)

mark_as_advanced(
  TreeSitter_INCLUDE_DIR
  TreeSitter_LIBRARY
  TreeSitter_BINARY
)
