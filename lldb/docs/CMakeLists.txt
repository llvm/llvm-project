include(FindDoxygen)

if(DOXYGEN_FOUND)
  set(abs_top_srcdir ${CMAKE_CURRENT_SOURCE_DIR}/..)
  set(DOT dot)
  set(PACKAGE_VERSION mainline)
  set(abs_top_builddir ..)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)

  add_custom_target(lldb-cpp-doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating LLDB C++ API reference with Doxygen" VERBATIM
  )
endif()

if (NOT LLDB_DISABLE_PYTHON)
  find_program(EPYDOC_EXECUTABLE NAMES epydoc epydoc.py)
  if(EPYDOC_EXECUTABLE)
    message(STATUS "Found epydoc - ${EPYDOC_EXECUTABLE}")

    find_program(DOT_EXECUTABLE dot)
    if(DOT_EXECUTABLE)
      set(EPYDOC_OPTIONS ${EPYDOC_OPTIONS} --graph all --dotpath ${DOT_EXECUTABLE})
      message(STATUS "Found dot - ${DOT_EXECUTABLE}")
    endif()

    # Pretend to make a python package so that we can generate the reference.
    # Because we don't build liblldb, epydoc will complain that the import of
    # _lldb.so failed, but that doesn't prevent it from generating the docs.
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lldb)
    get_target_property(lldb_scripts_dir swig_wrapper BINARY_DIR)
    add_custom_target(lldb-python-doc-package
      COMMAND "${CMAKE_COMMAND}" -E copy "${lldb_scripts_dir}/lldb.py" "${CMAKE_CURRENT_BINARY_DIR}/lldb/__init__.py"
      COMMENT "Copying lldb.py to pretend package.")
    add_dependencies(lldb-python-doc-package swig_wrapper)

    set(DOC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/doc")
    file(MAKE_DIRECTORY "${DOC_DIR}")
    add_custom_target(lldb-python-doc
      ${EPYDOC_EXECUTABLE}
      --html
      lldb
      -o ${CMAKE_CURRENT_BINARY_DIR}/python_reference
      --name "LLDB python API"
      --url "http://lldb.llvm.org"
      ${EPYDOC_OPTIONS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating LLDB Python API reference with epydoc" VERBATIM
    )
    add_dependencies(lldb-python-doc swig_wrapper lldb-python-doc-package)
  else()
    message(STATUS "Could NOT find epydoc")
  endif()
endif ()

if (LLVM_ENABLE_SPHINX)
  include(AddSphinxTarget)
  if (SPHINX_FOUND)
    if (${SPHINX_OUTPUT_HTML})
      add_sphinx_target(html lldb)
      # Sphinx does not reliably update the custom CSS files, so force
      # a clean rebuild of the documentation every time.
      add_custom_target(clean-lldb-html COMMAND "${CMAKE_COMMAND}" -E
        remove_directory ${CMAKE_CURRENT_BINARY_DIR}/html)
      add_dependencies(docs-lldb-html clean-lldb-html)
    endif()

    if (${SPHINX_OUTPUT_MAN})
      add_sphinx_target(man lldb)
    endif()
  endif()
endif()
