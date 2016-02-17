# Needed for lit support
include(AddLLVM)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.common.unit.configured.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.common.unit.configured)
