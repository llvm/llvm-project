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

if(CLANG_BOLT_INSTRUMENT AND NOT LLVM_BUILD_INSTRUMENTED)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/bolt.lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/bolt-fdata/lit.site.cfg
    )

  add_lit_testsuite(generate-bolt-fdata "Generating BOLT profile for Clang"
    ${CMAKE_CURRENT_BINARY_DIR}/bolt-fdata/
    EXCLUDE_FROM_CHECK_ALL
    DEPENDS clang-instrumented clear-bolt-fdata
    )

  add_custom_target(clear-bolt-fdata
    COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py clean ${CMAKE_CURRENT_BINARY_DIR} fdata
    COMMENT "Clearing old BOLT fdata")

  # Merge profiles into one using merge-fdata
  add_custom_target(clang-bolt-profile
    COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/perf-helper.py merge-fdata $<TARGET_FILE:merge-fdata> ${CMAKE_CURRENT_BINARY_DIR}/prof.fdata ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Merging BOLT fdata"
    DEPENDS merge-fdata generate-bolt-fdata)
endif()
