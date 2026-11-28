add_llvm_library(23cnt7_ksenia_igonina_lab1 MODULE
  VariablePrefixer.cpp
  PLUGIN_TOOL
  clang
)

target_compile_options(23cnt7_ksenia_igonina_lab1 PRIVATE
  -fvisibility=hidden
  -fvisibility-inlines-hidden
)

set_target_properties(23cnt7_ksenia_igonina_lab1 PROPERTIES
  LINK_FLAGS "-Wl,--exclude-libs=ALL"
)

target_link_libraries(23cnt7_ksenia_igonina_lab1 PRIVATE
  clangASTMatchers
  clangRewrite
)

add_lit_testsuite(check-23cnt7 "Running lit tests for 23cnt7 plugin"
  ${CMAKE_CURRENT_SOURCE_DIR}/test
)
