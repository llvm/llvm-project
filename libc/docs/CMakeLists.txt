
if (LLVM_ENABLE_SPHINX)
include(AddSphinxTarget)
if (SPHINX_FOUND)
  if (${SPHINX_OUTPUT_HTML})
    add_sphinx_target(html libc)

    list(APPEND docgen_list
      assert
      ctype
      errno
      fenv
      float
      inttypes
      locale
      setjmp
      signal
      stdbit
      stdlib
      string
      strings
      threads
      uchar
      wchar
      wctype
    )

    foreach(stem IN LISTS docgen_list)
      add_custom_target(${stem}_rst
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../utils/docgen/docgen.py ${stem}.h >
                ${CMAKE_CURRENT_SOURCE_DIR}/headers/${stem}.rst)
      add_dependencies(docs-libc-html ${stem}_rst)
    endforeach()
  endif()
endif()
endif()
