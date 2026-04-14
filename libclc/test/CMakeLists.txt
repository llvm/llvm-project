set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

set(LIBCLC_TEST_DEPS
  libclc-opencl-builtins
)

umbrella_lit_testsuite_begin(check-libclc)

# Testing unresolved symbols.
# Skip nvptx, clspv, spirv targets
if(ARCH MATCHES amdgcn)
  # Get the output file from the target property
  set(target_file "$<TARGET_PROPERTY:libclc-${LIBCLC_TARGET},TARGET_FILE>")

  set(LIBCLC_TARGET_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/${LIBCLC_TARGET})
  file(MAKE_DIRECTORY ${LIBCLC_TARGET_TEST_DIR})
  file(GENERATE OUTPUT ${LIBCLC_TARGET_TEST_DIR}/check-external-funcs.test
    CONTENT "; RUN: llvm-nm -u \"${target_file}\" | FileCheck %s --allow-empty\n\n; CHECK-NOT: {{.+}}\n"
  )

  configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${LIBCLC_TARGET_TEST_DIR}/lit.site.cfg.py
      MAIN_CONFIG
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

  add_lit_testsuite(check-libclc-external-funcs-${LIBCLC_TARGET} "Running ${LIBCLC_TARGET} tests"
    ${LIBCLC_TARGET_TEST_DIR}
    DEPENDS ${LIBCLC_TEST_DEPS}
  )
  set_target_properties(check-libclc-external-funcs-${LIBCLC_TARGET} PROPERTIES FOLDER "libclc tests")
endif()

umbrella_lit_testsuite_end(check-libclc)
