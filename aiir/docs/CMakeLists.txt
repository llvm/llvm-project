
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
    set(AIIR_DOXYGEN_QCH_FILENAME "org.llvm.aiir.qch" CACHE STRING
      "Filename of the Qt Compressed help file")
    set(AIIR_DOXYGEN_QHP_NAMESPACE "org.llvm.aiir" CACHE STRING
      "Namespace under which the intermediate Qt Help Project file lives")
    set(AIIR_DOXYGEN_QHP_CUST_FILTER_NAME "AIIR ${AIIR_VERSION}" CACHE STRING
      "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#custom-filters")
    set(AIIR_DOXYGEN_QHP_CUST_FILTER_ATTRS "AIIR,${AIIR_VERSION}" CACHE STRING
      "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#filter-attributes")
    set(aiir_doxygen_generate_qhp "YES")
    set(aiir_doxygen_qch_filename "${AIIR_DOXYGEN_QCH_FILENAME}")
    set(aiir_doxygen_qhp_namespace "${AIIR_DOXYGEN_QHP_NAMESPACE}")
    set(aiir_doxygen_qhelpgenerator_path "${LLVM_DOXYGEN_QHELPGENERATOR_PATH}")
    set(aiir_doxygen_qhp_cust_filter_name "${AIIR_DOXYGEN_QHP_CUST_FILTER_NAME}")
    set(aiir_doxygen_qhp_cust_filter_attrs "${AIIR_DOXYGEN_QHP_CUST_FILTER_ATTRS}")
  else()
    set(aiir_doxygen_generate_qhp "NO")
    set(aiir_doxygen_qch_filename "")
    set(aiir_doxygen_qhp_namespace "")
    set(aiir_doxygen_qhelpgenerator_path "")
    set(aiir_doxygen_qhp_cust_filter_name "")
    set(aiir_doxygen_qhp_cust_filter_attrs "")
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
  set(aiir_doxygen_generate_qhp)
  set(aiir_doxygen_qch_filename)
  set(aiir_doxygen_qhp_namespace)
  set(aiir_doxygen_qhelpgenerator_path)
  set(aiir_doxygen_qhp_cust_filter_name)
  set(aiir_doxygen_qhp_cust_filter_attrs)

  add_custom_target(doxygen-aiir
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating aiir doxygen documentation." VERBATIM)
  set_target_properties(doxygen-aiir PROPERTIES FOLDER "AIIR/Docs")

  if (LLVM_BUILD_DOCS)
    add_dependencies(doxygen doxygen-aiir)
  endif()

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
      DESTINATION docs/html)
  endif()
endif()
endif()

