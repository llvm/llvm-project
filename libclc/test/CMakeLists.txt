set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

set(LIBCLC_TEST_DEPS
  libclc-opencl-builtins
)

umbrella_lit_testsuite_begin(check-libclc)

# Testing unresolved symbols.
foreach(t ${LIBCLC_TARGETS_TO_BUILD})
  string(REPLACE "-" ";" TRIPLE ${t})
  list(GET TRIPLE 0 ARCH)

  # Skip nvptx, clspv, spirv targets
  if(ARCH MATCHES "^(nvptx|clspv)(64)?$" OR ARCH MATCHES "^spirv(64)?$")
    continue()
  endif()

  # Get the output file from the target property
  set(target_file "$<TARGET_PROPERTY:libclc-${t},TARGET_FILE>")

  set(LIBCLC_TARGET_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/${t})
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

  add_lit_testsuite(check-libclc-external-funcs-${t} "Running ${t} tests"
    ${LIBCLC_TARGET_TEST_DIR}
    DEPENDS ${LIBCLC_TEST_DEPS}
  )
  set_target_properties(check-libclc-external-funcs-${t} PROPERTIES FOLDER "libclc tests")
endforeach()

umbrella_lit_testsuite_end(check-libclc)
