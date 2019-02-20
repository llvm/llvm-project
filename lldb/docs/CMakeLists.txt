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
endif(DOXYGEN_FOUND)

find_package(PythonInterp REQUIRED)
find_program(EPYDOC_EXECUTABLE NAMES epydoc epydoc.py)
if(EPYDOC_EXECUTABLE)
  find_program(DOT_EXECUTABLE dot)
    if(DOT_EXECUTABLE)
      set(EPYDOC_OPTIONS ${EPYDOC_OPTIONS} --graph all --dotpath ${DOT_EXECUTABLE})
    endif()
    set(DOC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/doc")
    file(MAKE_DIRECTORY "${DOC_DIR}")
    #set(ENV{PYTHONPATH} ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/python2.7/site-packages)
    add_custom_target(lldb-python-doc
      ${EPYDOC_EXECUTABLE}
      --html
      lldb
      -o ${CMAKE_CURRENT_BINARY_DIR}/python_reference
      --name "LLDB python API"
      --url "http://lldb.llvm.org"
      ${EPYDOC_OPTIONS}
      DEPENDS swig_wrapper liblldb
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../../lib${LLVM_LIBDIR_SUFFIX}/python2.7/site-packages
      COMMENT "Generating LLDB Python API reference with epydoc" VERBATIM
    )
endif(EPYDOC_EXECUTABLE)

if (LLVM_ENABLE_SPHINX)
  include(AddSphinxTarget)
  if (SPHINX_FOUND)
    if (${SPHINX_OUTPUT_HTML})
      add_sphinx_target(html lldb)
    endif()
  endif()
endif()
