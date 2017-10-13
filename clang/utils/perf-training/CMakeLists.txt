
# All test suites added here should be excuded from check-all
set(EXCLUDE_FROM_ALL On)

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} CLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

if(LLVM_BUILD_INSTRUMENTED)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/pgo-data/lit.site.cfg
    )

  add_lit_testsuite(generate-profraw "Generating clang PGO data"
    ${CMAKE_CURRENT_BINARY_DIR}/pgo-data/
    DEPENDS clang clear-profraw
    )

  add_custom_target(clear-profraw
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py clean ${CMAKE_CURRENT_BINARY_DIR} profraw
    COMMENT "Clearing old profraw data")

  if(NOT LLVM_PROFDATA)
    find_program(LLVM_PROFDATA llvm-profdata)
  endif()

  if(NOT LLVM_PROFDATA)
    message(STATUS "To enable merging PGO data LLVM_PROFDATA has to point to llvm-profdata")
  else()
    add_custom_target(generate-profdata
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py merge ${LLVM_PROFDATA} ${CMAKE_CURRENT_BINARY_DIR}/clang.profdata ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Merging profdata"
      DEPENDS generate-profraw)
  endif()
endif()

find_program(DTRACE dtrace)
if(APPLE AND DTRACE)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/order-files.lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/order-files/lit.site.cfg
    )

  add_lit_testsuite(generate-dtrace-logs "Generating clang dtrace data"
    ${CMAKE_CURRENT_BINARY_DIR}/order-files/
    ARGS -j 1
    DEPENDS clang clear-dtrace-logs
    )

  add_custom_target(clear-dtrace-logs
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py clean ${CMAKE_CURRENT_BINARY_DIR} dtrace
    COMMENT "Clearing old dtrace data")

  if(NOT CLANG_ORDER_FILE)
    message(FATAL_ERROR "Output clang order file is not set")
  endif()

  add_custom_target(generate-order-file
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py gen-order-file --binary $<TARGET_FILE:clang> --output ${CLANG_ORDER_FILE} ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating order file"
    DEPENDS generate-dtrace-logs)
endif()
