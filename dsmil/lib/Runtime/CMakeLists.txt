set(DSMIL_RUNTIME_SOURCES
  dsmil_stealth_runtime.c
  dsmil_radio_runtime.c
  dsmil_nuclear_surety_runtime.c
  dsmil_mpe_runtime.c
  dsmil_jadc2_runtime.c
  dsmil_edge_security_runtime.c
  dsmil_cross_domain_runtime.c
  dsmil_blue_red_runtime.c
  dsmil_bft_runtime.c
)

add_library(dsmilrt STATIC ${DSMIL_RUNTIME_SOURCES})
set_target_properties(dsmilrt PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(dsmilrt PUBLIC
  ${LLVM_MAIN_SRC_DIR}/../dsmil/include
)

install(TARGETS dsmilrt
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
