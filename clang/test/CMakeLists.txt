# Test runner infrastructure for Clang. This configures the Clang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.
#
# If this is a stand-alone Clang build, we fake up our own Lit support here
# rather than relying on LLVM's.

set(CLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CLANG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

if( PATH_TO_LLVM_BUILD )
  set(CLANG_TEST_EXTRA_ARGS "--path=${CLANG_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}")
endif()

option(CLANG_TEST_USE_VG "Run Clang tests under Valgrind" OFF)
if(CLANG_TEST_USE_VG)
  set(CLANG_TEST_EXTRA_ARGS ${CLANG_TEST_EXTRA_ARGS} "--vg")
endif ()

if( NOT CLANG_BUILT_STANDALONE )

  set(CLANG_TEST_DEPS
    clang clang-headers
    c-index-test diagtool arcmt-test c-arcmt-test
    clang-check
    llvm-dis llc opt FileCheck count not
    )
  set(CLANG_TEST_PARAMS
    clang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    )

  if(LLVM_INCLUDE_TESTS)
    list(APPEND CLANG_TEST_DEPS ClangUnitTests)
    list(APPEND CLANG_TEST_PARAMS
      clang_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
      )
  endif()
  add_lit_testsuite(check-clang "Running the Clang regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${CLANG_TEST_PARAMS}
    DEPENDS ${CLANG_TEST_DEPS}
    ARGS ${CLANG_TEST_EXTRA_ARGS}
    )
  set_target_properties(check-clang PROPERTIES FOLDER "Clang tests")

else()

  include(FindPythonInterp)
  if(PYTHONINTERP_FOUND)
    if( LLVM_MAIN_SRC_DIR )
      set(LIT "${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py")
    else()
      set(LIT "${PATH_TO_LLVM_BUILD}/bin/${CMAKE_CFG_INTDIR}/llvm-lit")
      # Installed LLVM does not contain ${CMAKE_CFG_INTDIR} in paths.
      if( NOT EXISTS ${LIT} )
        set(LIT "${PATH_TO_LLVM_BUILD}/bin/llvm-lit")
      endif()
    endif()

    set(LIT_ARGS "${CLANG_TEST_EXTRA_ARGS} ${LLVM_LIT_ARGS}")
    separate_arguments(LIT_ARGS)

    add_custom_target(check-clang
      COMMAND ${PYTHON_EXECUTABLE}
              ${LIT}
              --param clang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
              --param build_config=${CMAKE_CFG_INTDIR}
              --param build_mode=${RUNTIME_BUILD_MODE}
              ${LIT_ARGS}
              ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Running Clang regression tests"
      DEPENDS clang clang-headers
              c-index-test diagtool arcmt-test c-arcmt-test
              clang-check
      )
    set_target_properties(check-clang PROPERTIES FOLDER "Clang tests")
  endif()

endif()

# Add a legacy target spelling: clang-test
add_custom_target(clang-test)
add_dependencies(clang-test check-clang)
set_target_properties(clang-test PROPERTIES FOLDER "Clang tests")
