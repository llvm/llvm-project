#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(ARCH_DEP_FILES
  ${TARGET_ARCHITECTURE}/ll_abi.cpp
)

set_source_files_properties(${UTILS_SYMTAB_BIN_DIR}/symtabdf.cpp PROPERTIES GENERATED TRUE)

set(SOURCES
  ${ARCH_DEP_FILES}
  asprintf.cpp
  bihutil.cpp
  dinitutl.cpp
  dinit.cpp
  dtypeutl.cpp
  fenddf.cpp
  ilmutil.cpp
  listing.cpp
  mwd.cpp
  semsym.cpp
  semutil0.cpp
  symtab.cpp
  symacc.cpp
  upper.cpp
  xref.cpp
  expreg.cpp
  exputil.cpp
  regutil.cpp
  expdf.cpp
  ili-rewrite.cpp
  ilidf.cpp
  iliutil.cpp
  iltutil.cpp
  expand.cpp
  expatomics.cpp
  expsmp.cpp
  exp_ftn.cpp
  exp_fvec.cpp
  exp_rte.cpp
  outliner.cpp
  rmsmove.cpp
  scope.cpp
  rbtree.cpp
  dwarf_names.cpp
  llassem.cpp
  llassem_common.cpp
  asm_anno.cpp
  main.cpp
  version.cpp
  cgmain.cpp
  fastset.cpp
  lldebug.cpp
  llutil.cpp
  ll_ftn.cpp
  ll_structure.cpp
  ll_write.cpp
  ll_builder.cpp
  llopt.cpp
  llsched.cpp
  machreg.cpp
  kmpcutil.cpp
  verify.cpp
  ompaccel.cpp
  tgtutil.cpp
  ${UTILS_SYMTAB_BIN_DIR}/symtabdf.cpp
  )

set(COMMON_DEFS
  MMD
  NOVECTORIZE
  IN_FLANG2
  )

set(INCLUDE_DIRS
  ${TOOLS_SHARED_DIR}
  ${FLANG_INCLUDE_DIR}
  ${FLANG_SOURCE_DIR}/lib/scutil
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_ARCHITECTURE}
  ${UTILS_SYMTAB_BIN_DIR}        # Symbol table headers
  ${UTILS_ILI_BIN_DIR}           # ILI IR headers
  ${UTILS_ILM_BIN_DIR}           # ILM IR headers
  ${UTILS_MACHAR_BIN_DIR}        # ILM IR headers
  ${UTILS_UPPER_BIN_DIR}         # `Upper' headers
  ${UTILS_SHARED_DIR}
  )

set(SHARED_SOURCES
  ${SHARED_SOURCES}
  ${LIB_SOURCE_DIR}/ADT/hash.c
  ${LIB_SOURCE_DIR}/ArgParser/arg_parser.c
  ${LIB_SOURCE_DIR}/ArgParser/debug_action.c
  ${LIB_SOURCE_DIR}/ArgParser/xflag.c
  )


foreach(FILE ${SHARED_SOURCES})
  get_filename_component(TOFILE ${FILE} NAME)
  if(WIN32)
    add_custom_command(OUTPUT ${TOFILE}pp
      COMMAND ${CMAKE_COMMAND} -E copy ${FILE} ${TOFILE}pp
      MAIN_DEPENDENCY ${FILE})
  else()
    add_custom_command(OUTPUT ${TOFILE}pp
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${FILE} ${TOFILE}pp
      MAIN_DEPENDENCY ${FILE})
  endif()
  set(SHARED_CPP_SOURCES ${SHARED_CPP_SOURCES} ${TOFILE}pp)
endforeach()

add_flang_executable(flang2
  ${SOURCES} ${SHARED_CPP_SOURCES}
  )

target_compile_definitions(flang2
  PRIVATE
  ${COMMON_DEFS}
  )

target_compile_options(flang2
  PRIVATE
  ${COMPILE_OPTS}
  )

target_link_libraries(flang2
  PRIVATE
  flangArgParser
  scutil
  )

if (NOT MSVC)
  target_link_libraries(flang2 PRIVATE m)
endif()

add_dependencies(flang2
  gen_backend_error_headers # Error message headers
  gen_backend_symtab        # Symbol table headers
  gen_backend_symini        # Symbol table (symini) headers
  gen_backend_ili           # ILI IR headers
  gen_backend_ilm           # ILM IR headers
  gen_backend_machar        # Machar headers
  gen_backend_upper         # `Upper' headers
  )

if (FLANG_INCLUDE_DOCS)
  add_dependencies(flang2
    flang2_gen_sphinx_docs
    )
endif()

target_include_directories(flang2
  PRIVATE
  ${INCLUDE_DIRS}
  )

# Install flang2 executable
install(TARGETS flang2
        RUNTIME DESTINATION bin)

# Local Variables:
# mode: cmake
# End:
