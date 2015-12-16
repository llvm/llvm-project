if(LLVM_BUILD_INSTRUMENTED)
  if (CMAKE_CFG_INTDIR STREQUAL ".")
    set(LLVM_BUILD_MODE ".")
  else ()
    set(LLVM_BUILD_MODE "%(build_mode)s")
  endif ()

  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} CLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    )

  add_lit_testsuite(generate-profraw "Generating clang PGO data"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS clang clear-profraw
    )

  add_custom_target(clear-profraw
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py clean ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Clearing old profraw data")

  if(NOT LLVM_PROFDATA)
    find_program(LLVM_PROFDATA llvm-profdata)
  endif()

  if(NOT LLVM_PROFDATA)
    message(FATAL_ERROR "Must set LLVM_PROFDATA to point to llvm-profdata to use for merging PGO data")
  endif()

  add_custom_target(generate-profdata
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py merge ${LLVM_PROFDATA} ${CMAKE_CURRENT_BINARY_DIR}/clang.profdata ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Merging profdata"
    DEPENDS generate-profraw)
endif()
