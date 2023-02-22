set(CLANG_PGO_TRAINING_DATA "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
  "The path to a lit testsuite containing samples for PGO and order file generation"
  )

if(LLVM_BUILD_INSTRUMENTED)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/pgo-data/lit.site.cfg
    )

  add_lit_testsuite(generate-profraw "Generating clang PGO data"
    ${CMAKE_CURRENT_BINARY_DIR}/pgo-data/
    EXCLUDE_FROM_CHECK_ALL
    DEPENDS clang clear-profraw ${CLANG_PERF_TRAINING_DEPS}
    )

  add_custom_target(clear-profraw
    COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py clean ${CMAKE_CURRENT_BINARY_DIR} profraw
    COMMENT "Clearing old profraw data")

  if(NOT LLVM_PROFDATA)
    find_program(LLVM_PROFDATA llvm-profdata)
  endif()

  if(NOT LLVM_PROFDATA)
    message(STATUS "To enable merging PGO data LLVM_PROFDATA has to point to llvm-profdata")
  else()
    add_custom_target(generate-profdata
      COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py merge ${LLVM_PROFDATA} ${CMAKE_CURRENT_BINARY_DIR}/clang.profdata ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Merging profdata"
      DEPENDS generate-profraw)
  endif()
endif()

find_program(DTRACE dtrace)
# TODO: Look into supporting this for the driver build. It will require changing
# the perf-helper.py file to understand to call `llvm` as `llvm clang`.
if(APPLE AND DTRACE AND NOT LLVM_TOOL_LLVM_DRIVER_BUILD)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/order-files.lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/order-files/lit.site.cfg
    )

  add_lit_testsuite(generate-dtrace-logs "Generating clang dtrace data"
    ${CMAKE_CURRENT_BINARY_DIR}/order-files/
    EXCLUDE_FROM_CHECK_ALL
    ARGS -j 1
    DEPENDS clang clear-dtrace-logs
    )

  add_custom_target(clear-dtrace-logs
    COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py clean ${CMAKE_CURRENT_BINARY_DIR} dtrace
    COMMENT "Clearing old dtrace data")

  if(NOT CLANG_ORDER_FILE)
    message(FATAL_ERROR "Output clang order file is not set")
  endif()

  add_custom_target(generate-order-file
    COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py gen-order-file --binary $<TARGET_FILE:clang> --output ${CLANG_ORDER_FILE} ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating order file"
    DEPENDS generate-dtrace-logs)
endif()
