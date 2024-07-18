set(SAFESTACK_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SAFESTACK_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(SAFESTACK_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND SAFESTACK_TEST_DEPS safestack)
if(NOT COMPILER_RT_STANDALONE_BUILD)
  # Some tests require LTO, so add a dependency on the relevant LTO plugin.
  if(LLVM_ENABLE_PIC)
    if(LLVM_BINUTILS_INCDIR)
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
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  )

add_lit_testsuite(check-safestack "Running the SafeStack tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SAFESTACK_TEST_DEPS})
