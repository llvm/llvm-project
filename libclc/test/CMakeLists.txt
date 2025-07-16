set( LIBCLC_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )

set( LIBCLC_TEST_TARGETS_ALL
  amdgcn-mesa-mesa3d
)

foreach( target IN LISTS LIBCLC_TEST_TARGETS_ALL )
  # If we haven't built this libclc target, don't build the tests
  if( NOT TARGET prepare-${target} )
    message( WARNING "libclc tests require target ${target}. Tests will not be built" )
    # Add a dummy target
    add_custom_target( check-libclc )
    return()
  endif()

  list( APPEND LIBCLC_TEST_DEPS prepare-${target} )
endforeach()

list( APPEND LIBCLC_TEST_DEPS
  ${clang_target}
  FileCheck
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite( check-libclc
  "Running libclc regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LIBCLC_TEST_DEPS}
)
