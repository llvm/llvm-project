set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

if(ARCH STREQUAL amdgcn)
  set(LIBCLC_CPU "gfx900")
else()
  set(LIBCLC_CPU "")
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-libclc "Running libclc tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS libclc
)
set_target_properties(check-libclc PROPERTIES FOLDER "libclc tests")
