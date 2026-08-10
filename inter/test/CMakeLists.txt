configure_file(lit.site.cfg.py.in lit.cfg.py @ONLY)

add_executable(inter-ged-dump ged-dump.cpp)
target_link_libraries(inter-ged-dump PRIVATE InterGED)

add_custom_target(check-inter
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-lit -sv ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS inter-ged-dump inter-opt inter-translate
  USES_TERMINAL)
