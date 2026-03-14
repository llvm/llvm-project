set(LIBCLC_LIBRARY_DIR ${LIBCLC_OUTPUT_LIBRARY_DIR})
set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

set(LIBCLC_TEST_DEPS
  llvm-dis
)

umbrella_lit_testsuite_begin(check-libclc)

# Testing unresolved symbols.
# check_external_funcs.sh does not work on Windows
if(NOT WIN32)
  foreach(t ${LIBCLC_TARGETS_TO_BUILD})
    string(REPLACE "-" ";" TRIPLE ${t})
    list(GET TRIPLE 0 ARCH)

    # Skip nvptx, clspv, spirv targets
    if(ARCH MATCHES "^(nvptx|clspv)(64)?$" OR ARCH MATCHES "^spirv(64)?$")
      continue()
    endif()

    # Get the output file from the target property
    get_target_property(target_file libclc-${t} TARGET_FILE)
    get_filename_component(output_file ${target_file} NAME)

    set(LIBCLC_TARGET_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/${t})
    file(MAKE_DIRECTORY ${LIBCLC_TARGET_TEST_DIR})
    file(WRITE ${LIBCLC_TARGET_TEST_DIR}/check-external-funcs.test
"// RUN: %check_external_funcs %libclc_library_dir/${t}/${output_file} %llvm_tools_dir

// This test verifies that the libclc library for ${t} has no
// unresolved external functions (except LLVM intrinsics).
")

    configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${LIBCLC_TARGET_TEST_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
    )

    add_lit_testsuite(check-libclc-external-funcs-${t} "Running ${t} tests"
      ${LIBCLC_TARGET_TEST_DIR}
      DEPENDS libclc-${t} ${LIBCLC_TEST_DEPS}
    )
    set_target_properties(check-libclc-external-funcs-${t} PROPERTIES FOLDER "libclc tests")
  endforeach()
endif()

umbrella_lit_testsuite_end(check-libclc)
