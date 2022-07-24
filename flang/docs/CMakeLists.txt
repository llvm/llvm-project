
find_package(Doxygen)
if (DOXYGEN_FOUND)
if (LLVM_ENABLE_DOXYGEN)
  set(abs_top_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
  set(abs_top_builddir ${CMAKE_CURRENT_BINARY_DIR})

  if (HAVE_DOT)
    set(DOT ${LLVM_PATH_DOT})
  endif()

  if (LLVM_DOXYGEN_EXTERNAL_SEARCH)
    set(enable_searchengine "YES")
    set(searchengine_url "${LLVM_DOXYGEN_SEARCHENGINE_URL}")
    set(enable_server_based_search "YES")
    set(enable_external_search "YES")
    set(extra_search_mappings "${LLVM_DOXYGEN_SEARCH_MAPPINGS}")
  else()
    set(enable_searchengine "NO")
    set(searchengine_url "")
    set(enable_server_based_search "NO")
    set(enable_external_search "NO")
    set(extra_search_mappings "")
  endif()

  # If asked, configure doxygen for the creation of a Qt Compressed Help file.
  if (LLVM_ENABLE_DOXYGEN_QT_HELP)
    set(FLANG_DOXYGEN_QCH_FILENAME "org.llvm.flang.qch" CACHE STRING
      "Filename of the Qt Compressed help file")
    set(FLANG_DOXYGEN_QHP_NAMESPACE "org.llvm.flang" CACHE STRING
      "Namespace under which the intermediate Qt Help Project file lives")
    set(FLANG_DOXYGEN_QHP_CUST_FILTER_NAME "FLANG ${FLANG_VERSION}" CACHE STRING
      "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#custom-filters")
    set(FLANG_DOXYGEN_QHP_CUST_FILTER_ATTRS "FLANG,${FLANG_VERSION}" CACHE STRING
      "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#filter-attributes")
    set(flang_doxygen_generate_qhp "YES")
    set(flang_doxygen_qch_filename "${FLANG_DOXYGEN_QCH_FILENAME}")
    set(flang_doxygen_qhp_namespace "${FLANG_DOXYGEN_QHP_NAMESPACE}")
    set(flang_doxygen_qhelpgenerator_path "${LLVM_DOXYGEN_QHELPGENERATOR_PATH}")
    set(flang_doxygen_qhp_cust_filter_name "${FLANG_DOXYGEN_QHP_CUST_FILTER_NAME}")
    set(flang_doxygen_qhp_cust_filter_attrs "${FLANG_DOXYGEN_QHP_CUST_FILTER_ATTRS}")
  else()
    set(flang_doxygen_generate_qhp "NO")
    set(flang_doxygen_qch_filename "")
    set(flang_doxygen_qhp_namespace "")
    set(flang_doxygen_qhelpgenerator_path "")
    set(flang_doxygen_qhp_cust_filter_name "")
    set(flang_doxygen_qhp_cust_filter_attrs "")
  endif()

  option(LLVM_DOXYGEN_SVG
    "Use svg instead of png files for doxygen graphs." OFF)
  if (LLVM_DOXYGEN_SVG)
    set(DOT_IMAGE_FORMAT "svg")
  else()
    set(DOT_IMAGE_FORMAT "png")
  endif()

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)

  set(abs_top_srcdir)
  set(abs_top_builddir)
  set(DOT)
  set(enable_searchengine)
  set(searchengine_url)
  set(enable_server_based_search)
  set(enable_external_search)
  set(extra_search_mappings)
  set(flang_doxygen_generate_qhp)
  set(flang_doxygen_qch_filename)
  set(flang_doxygen_qhp_namespace)
  set(flang_doxygen_qhelpgenerator_path)
  set(flang_doxygen_qhp_cust_filter_name)
  set(flang_doxygen_qhp_cust_filter_attrs)
  set(DOT_IMAGE_FORMAT)

  add_custom_target(doxygen-flang
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating flang doxygen documentation." VERBATIM)

  if (LLVM_BUILD_DOCS)
   add_dependencies(doxygen doxygen-flang)
  endif()

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY AND LLVM_BUILD_DOCS)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
      DESTINATION docs/html)
  endif()
endif()
endif()

function (gen_rst_file_from_td output_file td_option source docs_target)
  if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    message(FATAL_ERROR "Cannot find source file: ${source} in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  get_filename_component(TABLEGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${source}" DIRECTORY)
  list(APPEND LLVM_TABLEGEN_FLAGS "-I${TABLEGEN_INCLUDE_DIR}")
  clang_tablegen(Source/${output_file} ${td_option} SOURCE ${source} TARGET "gen-${output_file}")
  add_dependencies(${docs_target} "gen-${output_file}")
endfunction()

if (LLVM_ENABLE_SPHINX)
  include(AddSphinxTarget)
  if (SPHINX_FOUND)
    if (${SPHINX_OUTPUT_HTML})
      add_sphinx_target(html flang SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/Source")

      add_dependencies(docs-flang-html copy-flang-src-docs)

      # Copy the flang/docs directory and the generated FIRLangRef.md file to a place in the binary directory.
      # Having all the files in a single directory makes it possible for Sphinx to process them together.
      # Add a dependency to the flang-doc target to ensure that the FIRLangRef.md file is generated before the copying happens.
      add_custom_target(copy-flang-src-docs
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/Source"
        DEPENDS flang-doc)

      # Runs a python script prior to HTML generation to prepend a header to FIRLangRef,
      # Without the header, the page is incorrectly formatted, as it assumes the first entry is the page title.
      add_custom_command(TARGET copy-flang-src-docs
        COMMAND "${Python3_EXECUTABLE}"
        ARGS ${CMAKE_CURRENT_BINARY_DIR}/Source/FIR/CreateFIRLangRef.py)

      # CLANG_TABLEGEN_EXE variable needs to be set for clang_tablegen to run without error
      set(CLANG_TABLEGEN_EXE clang-tblgen)
      gen_rst_file_from_td(FlangCommandLineReference.rst -gen-opt-docs ../include/flang/FlangOptionsDocs.td docs-flang-html)
    endif()
    if (${SPHINX_OUTPUT_MAN})
      add_sphinx_target(man flang)
    endif()
  endif()
endif()

