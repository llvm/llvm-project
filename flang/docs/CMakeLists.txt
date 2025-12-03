
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

  add_custom_target(doxygen-flang
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating flang doxygen documentation." VERBATIM)
  set_target_properties(doxygen-flang PROPERTIES FOLDER "Flang/Docs")
  if (LLVM_BUILD_DOCS)
   add_dependencies(doxygen doxygen-flang)
  endif()

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY AND LLVM_BUILD_DOCS)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
      DESTINATION docs/html)
  endif()
endif()
endif()

function (gen_rst_file_from_td output_file td_option source target)
  if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    message(FATAL_ERROR "Cannot find source file: ${source} in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  get_filename_component(TABLEGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${source}" DIRECTORY)
  list(APPEND LLVM_TABLEGEN_FLAGS "-I${TABLEGEN_INCLUDE_DIR}")
  list(APPEND LLVM_TABLEGEN_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../../clang/include/clang/Options/")
  clang_tablegen(Source/${output_file} ${td_option} SOURCE ${source} TARGET ${target})
endfunction()

if (LLVM_ENABLE_SPHINX)
  set (FLANG_DOCS_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/SourceHtml")
  set (FLANG_DOCS_MAN_DIR "${CMAKE_CURRENT_BINARY_DIR}/SourceMan")
  include(AddSphinxTarget)
  if (SPHINX_FOUND)

    # CLANG_TABLEGEN_EXE variable needs to be set for clang_tablegen to run without error
    find_program(CLANG_TABLEGEN_EXE "clang-tblgen" ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

    # Generate the RST file from TableGen (for both HTML and MAN builds)
    gen_rst_file_from_td(FlangCommandLineReference.rst -gen-opt-docs FlangOptionsDocs.td "gen-FlangCommandLineReference.rst")
    gen_rst_file_from_td(FlangCommandLineOptions.rst -gen-opt-docs FlangOptionsMan.td "gen-FlangCommandLineOptions.rst")
    # clang_tablegen() (called from gen_rst_file_from_td()) does not create the
    # output directory automatically, so we have to create it explicitly.
    add_custom_target(create-flang-rst-output-dir
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/Source
    )
    add_dependencies("gen-FlangCommandLineReference.rst" create-flang-rst-output-dir)
    add_dependencies("gen-FlangCommandLineOptions.rst" create-flang-rst-output-dir)

    if (${SPHINX_OUTPUT_HTML})
      message(STATUS "Using index.md for html build")

      # Copy the entire flang/docs directory to the build Source dir,
      # then remove the index.rst file, to avoid clash with index.md
      # which is used for the HTML build.
      add_custom_target(copy-flang-src-docs-html
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
              "${CMAKE_CURRENT_SOURCE_DIR}"
              "${FLANG_DOCS_HTML_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E remove
          "${FLANG_DOCS_HTML_DIR}/CommandGuide/index.rst"
        COMMAND "${CMAKE_COMMAND}" -E copy
          "${CMAKE_CURRENT_BINARY_DIR}/Source/FlangCommandLineReference.rst"
          "${FLANG_DOCS_HTML_DIR}/FlangCommandLineReference.rst"
        DEPENDS flang-doc gen-FlangCommandLineReference.rst)

      # Run Python preprocessing ONLY for HTML build
      # This script prepends headers to FIRLangRef.md for proper formatting
      add_custom_command(TARGET copy-flang-src-docs-html
        COMMAND "${Python3_EXECUTABLE}"
        ARGS "${FLANG_DOCS_HTML_DIR}/FIR/CreateFIRLangRef.py")

      add_sphinx_target(html flang SOURCE_DIR "${FLANG_DOCS_HTML_DIR}")
      add_dependencies(docs-flang-html copy-flang-src-docs-html)
    endif()

    # ----------------------------
    # MAN BUILD SETUP
    # ----------------------------
    if (${SPHINX_OUTPUT_MAN})
      message(STATUS "Using CommandGuide/index.rst for man build")

      # Create minimal Source dir with ONLY the files needed for man build:
      # - conf.py (Sphinx config)
      # - index.rst (top-level man page)
      # - FlangCommandLineOptions.rst (generated reference)
      add_custom_target(copy-flang-src-docs-man
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${FLANG_DOCS_MAN_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/conf.py"
          "${FLANG_DOCS_MAN_DIR}/conf.py"
        COMMAND "${CMAKE_COMMAND}" -E copy
          "${CMAKE_CURRENT_BINARY_DIR}/Source/FlangCommandLineOptions.rst"
          "${FLANG_DOCS_MAN_DIR}/FlangCommandLineOptions.rst"
        COMMAND "${CMAKE_COMMAND}" -E copy
                "${CMAKE_CURRENT_SOURCE_DIR}/CommandGuide/index.rst"
                "${FLANG_DOCS_MAN_DIR}/index.rst"
        DEPENDS flang-doc gen-FlangCommandLineOptions.rst)

      add_sphinx_target(man flang SOURCE_DIR "${FLANG_DOCS_MAN_DIR}")
      add_dependencies(docs-flang-man copy-flang-src-docs-man)
    endif()
  endif()
endif()

