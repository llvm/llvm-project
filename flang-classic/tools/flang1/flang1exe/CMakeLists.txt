#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set_source_files_properties(${UTILS_SYMTAB_BIN_DIR}/symtabdf.c PROPERTIES GENERATED TRUE)

set(SOURCES
  accpp.c
  assem.c
  ast.c
  astdf.c
  astout.c
  bblock.c
  comm.c
  commdf.c
  commgen.c
  comminvar.c
  commopt.c
  datadep.c
  detect.c
  dinit.c
  dinitutl.c
  dist.c
  dpm_out.c
  dtypeutl.c
  dummy.c
  dump.c
  exterf.c
  fenddf.c
  fgraph.c
  findloop.c
  flow.c
  flow_util.c
  fpp.c
  func.c
  hash.c
  hlvect.c
  hpfutl.c
  induc.c
  inliner.c
  interf.c
  invar.c
  iterat.c
  listing.c
  lower.c
  lowerchk.c
  lowerexp.c
  lowerilm.c
  lowersym.c
  lz.c
  main.c
  module.c
  optdf.c
  optimize.c
  optutil.c
  outconv.c
  parser.c
  pointsto.c
  psemant.c
  psemant2.c
  psemant3.c
  psemantio.c
  psemsmp.c
  pstride.c
  redundss.c
  rest.c
  rte.c
  scan.c
  scopestack.c
  semant.c
  semant2.c
  semant3.c
  semantio.c
  semast.c
  semfin.c
  semfunc.c
  semfunc2.c
  semgnr.c
  semsmp.c
  semsym.c
  semtbp.c
  semutil.c
  semutil2.c
  symacc.c
  symtab.c
  symutl.c
  transfrm.c
  version.c
  xref.c
  vsub.c
  ${UTILS_SYMTAB_BIN_DIR}/symtabdf.c
  )

set(COMMON_DEFS
  MMD
  PGF90
  )

set_source_files_properties(accpp.c PROPERTIES COMPILE_FLAGS -Wno-comment)
set_source_files_properties(fpp.c PROPERTIES COMPILE_FLAGS -Wno-comment)

set(INCLUDE_DIRS
  ${FLANG_INCLUDE_DIR}
  ${UTILS_SYMTAB_BIN_DIR}     # Symbol table and ILM headers
  ${UTILS_MACHAR_BIN_DIR}     # Machar headers
  ${UTILS_AST_BIN_DIR}        # AST headers
  ${UTILS_PARSETABLE_BIN_DIR} # Parser headers
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TOOLS_SHARED_DIR}
  )

if(NOT MSVC)
  set(COMPILE_OPTS
    -std=c11
    )
endif()

add_flang_executable(flang1
  ${SOURCES} ${SHARED_SOURCES}
  )

target_compile_definitions(flang1
  PRIVATE
  ${COMMON_DEFS}
  )

add_dependencies(flang1
  gen_frontend_symtab        # Symbol table
  gen_frontend_symini        # Symini (symbol table) ILM IR for lowering
  gen_frontend_machar        # Machar
  gen_frontend_ast           # AST
  gen_frontend_error_headers # Error message headers
  gen_frontend_parsetable    # Parser headers
  )

if (FLANG_INCLUDE_DOCS)
  file(MAKE_DIRECTORY ${FLANG1_DOC_BIN_DIR})
  add_dependencies(flang1
    flang1_gen_sphinx_docs
    )
  # Copy the rst files, that don't need to be generated with n2rst
  configure_file(${FLANG1_DOC_SRC_DIR}/howto_intrinsics.rst ${FLANG1_DOC_BIN_DIR}/howto_intrinsics.rst COPYONLY)
endif()

target_include_directories(flang1
  PRIVATE
  ${INCLUDE_DIRS}
  )

target_compile_options(flang1
  PRIVATE
  ${COMPILE_OPTS}
  )

target_link_libraries(flang1
  PRIVATE
  flangArgParser
  scutil
  )

if (NOT MSVC)
  target_link_libraries(flang1 PRIVATE m)
endif()

# Install flang1 executable
install(TARGETS flang1
  RUNTIME DESTINATION bin)

# Local Variables:
# mode: cmake
# End:
