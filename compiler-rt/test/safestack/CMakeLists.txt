set(SAFESTACK_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SAFESTACK_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(SAFESTACK_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND SAFESTACK_TEST_DEPS safestack)

  # Some tests require LTO, so add a dependency on the relevant LTO plugin.
  if(LLVM_ENABLE_PIC AND LLVM_BINUTILS_INCDIR)
    list(APPEND SAFESTACK_TEST_DEPS
      LLVMgold
    )
  endif()
  if(APPLE)
    list(APPEND SAFESTACK_TEST_DEPS
      LTO
    )
  endif()
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_lit_testsuite(check-safestack "Running the SafeStack tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SAFESTACK_TEST_DEPS})
set_target_properties(check-safestack PROPERTIES FOLDER "Compiler-RT Misc")
