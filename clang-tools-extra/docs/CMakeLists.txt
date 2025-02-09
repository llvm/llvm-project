if (DOXYGEN_FOUND)
  if (LLVM_ENABLE_DOXYGEN)
    set(abs_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
    set(abs_builddir ${CMAKE_CURRENT_BINARY_DIR})

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
      set(CLANG_TOOLS_DOXYGEN_QCH_FILENAME "org.llvm.clang.qch" CACHE STRING
        "Filename of the Qt Compressed help file")
      set(CLANG_TOOLS_DOXYGEN_QHP_NAMESPACE "org.llvm.clang" CACHE STRING
        "Namespace under which the intermediate Qt Help Project file lives")
      set(CLANG_TOOLS_DOXYGEN_QHP_CUST_FILTER_NAME "Clang ${CLANG_VERSION}" CACHE STRING
        "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#custom-filters")
      set(CLANG_TOOLS_DOXYGEN_QHP_CUST_FILTER_ATTRS "Clang,${CLANG_VERSION}" CACHE STRING
        "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#filter-attributes")
      set(clang_tools_doxygen_generate_qhp "YES")
      set(clang_tools_doxygen_qch_filename "${CLANG_DOXYGEN_QCH_FILENAME}")
      set(clang_tools_doxygen_qhp_namespace "${CLANG_DOXYGEN_QHP_NAMESPACE}")
      set(clang_tools_doxygen_qhelpgenerator_path "${LLVM_DOXYGEN_QHELPGENERATOR_PATH}")
      set(clang_tools_doxygen_qhp_cust_filter_name "${CLANG_DOXYGEN_QHP_CUST_FILTER_NAME}")
      set(clang_tools_doxygen_qhp_cust_filter_attrs "${CLANG_DOXYGEN_QHP_CUST_FILTER_ATTRS}")
    else()
      set(clang_tools_doxygen_generate_qhp "NO")
      set(clang_tools_doxygen_qch_filename "")
      set(clang_tools_doxygen_qhp_namespace "")
      set(clang_tools_doxygen_qhelpgenerator_path "")
      set(clang_tools_doxygen_qhp_cust_filter_name "")
      set(clang_tools_doxygen_qhp_cust_filter_attrs "")
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
    set(clang_tools_doxygen_generate_qhp)
    set(clang_tools_doxygen_qch_filename)
    set(clang_tools_doxygen_qhp_namespace)
    set(clang_tools_doxygen_qhelpgenerator_path)
    set(clang_tools_doxygen_qhp_cust_filter_name)
    set(clang_tools_doxygen_qhp_cust_filter_attrs)
    set(DOT_IMAGE_FORMAT)

    add_custom_target(doxygen-clang-tools
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating clang doxygen documentation." VERBATIM)
    set_target_properties(doxygen-clang-tools PROPERTIES FOLDER "Clang Tools Extra/Docs")

    if (LLVM_BUILD_DOCS)
      add_dependencies(doxygen doxygen-clang-tools)
    endif()

    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
      install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
        DESTINATION docs/html)
    endif()
  endif()
endif()

if (LLVM_ENABLE_SPHINX)
  include(AddSphinxTarget)
  if (SPHINX_FOUND)
    if (${SPHINX_OUTPUT_HTML})
      add_sphinx_target(html clang-tools)
    endif()
    if (${SPHINX_OUTPUT_MAN})
      add_sphinx_target(man clang-tools)
    endif()
  endif()
endif()
