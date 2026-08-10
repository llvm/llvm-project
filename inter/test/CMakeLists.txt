configure_file(lit.site.cfg.py.in lit.cfg.py @ONLY)

add_custom_target(check-inter
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-lit -sv ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS inter-opt inter-translate
  USES_TERMINAL)
