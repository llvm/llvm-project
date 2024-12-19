if (LLVM_ENABLE_SPHINX)
include(AddSphinxTarget)
if (SPHINX_FOUND)
  if (${SPHINX_OUTPUT_HTML})
    # Similar to clang, we copy our static .rst files from libc/docs/ to the
    # $build_dir/libc/docs/. That way, we can have a mix of both static
    # (committed) .rst files, and dynamically generated .rst files. We don't
    # want the dynamically generated .rst files to pollute the source tree.
    add_custom_target(copy-libc-rst-docs
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
      "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

    # Change sphinx to build from $build_dir/libc/docs/ rather than
    # llvm-project/libc/docs/.
    add_sphinx_target(html libc SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    # Depend on the copy target.
    add_dependencies(docs-libc-html copy-libc-rst-docs)

    # Maintain a list of headers for which we dynamically generate html docs
    # for via docgen. For more complex docs (such as per arch support, a la
    # math.h), those should be omitted and exist statically in
    # libc/docs/headers/.
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
      # docgen invocation.
      add_custom_target(${stem}_rst
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../utils/docgen/docgen.py ${stem}.h >
        ${CMAKE_CURRENT_BINARY_DIR}/headers/${stem}.rst)
      # depend on the docgen invocation.
      add_dependencies(docs-libc-html ${stem}_rst)
    endforeach()
  endif()
endif()
endif()
