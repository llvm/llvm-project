if (LLVM_ENABLE_SPHINX)
include(AddSphinxTarget)
if (SPHINX_FOUND)
  if (${SPHINX_OUTPUT_HTML})
    # Similar to clang, we copy our static documentation files from libc/docs/
    # to the $build_dir/libc/docs/. That way, we can have a mix of both static
    # committed files and dynamically generated .rst files. We don't want the
    # dynamically generated .rst files to pollute the source tree.
    # Maintain a list of headers for which we dynamically generate html docs
    # for via docgen. For more complex docs (such as per arch support, a la
    # math.h), those should be omitted and exist statically in
    # libc/docs/headers/.
    list(APPEND docgen_list
      aio
      arpa/inet
      assert
      byteswap
      cpio
      ctype
      dirent
      dlfcn
      endian
      errno
      fcntl
      fenv
      float
      glob
      inttypes
      libgen
      locale
      nl_types
      net/if
      netinet/in
      netinet/tcp
      poll
      pthread
      pwd
      sched
      setjmp
      signal
      spawn
      stdbit
      stdio
      stdlib
      string
      strings
      sys/ipc
      sys/mman
      sys/resource
      sys/select
      sys/sem
      sys/socket
      sys/stat
      sys/statvfs
      sys/time
      sys/uio
      sys/utsname
      sys/wait
      termios
      threads
      time
      uchar
      unistd
      wchar
      wctype
    )
    foreach(stem IN LISTS docgen_list)
      list(APPEND libc_generated_docs "headers/${stem}.rst")
    endforeach()
    list(APPEND libc_generated_docs configure.rst)
    add_sphinx_source_sync_target(copy-libc-rst-docs
      "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
      PRESERVE_DOCS ${libc_generated_docs})
    add_custom_command(TARGET copy-libc-rst-docs POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${CMAKE_CURRENT_SOURCE_DIR}/../Maintainers.md"
              "${CMAKE_CURRENT_BINARY_DIR}"
      )

    # For headers that are nested in directories, we need to
    # `mkdir $build_dir/libc/docs/headers/$dir`. Otherwise, the invocation of
    # docgen below will fail since the output file would be placed in a
    # directory that does not exist, leading to a `No such file or directory`
    # error from the shell.
    file(MAKE_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/headers/arpa/"
      "${CMAKE_CURRENT_BINARY_DIR}/headers/net/"
      "${CMAKE_CURRENT_BINARY_DIR}/headers/netinet/"
      "${CMAKE_CURRENT_BINARY_DIR}/headers/sys/"
    )

    # Change sphinx to build from $build_dir/libc/docs/ rather than
    # llvm-project/libc/docs/.
    add_sphinx_target(html libc SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    # Depend on the copy target.
    add_dependencies(docs-libc-html copy-libc-rst-docs)

    foreach(stem IN LISTS docgen_list)
      # It is an error in cmake to have a target name that contains a "/", but
      # docgen relies on the "/" to find headers nested under directories.
      # Replace with underscore and namespace the target to avoid collisions
      # with other runtime targets in combined builds.
      string(REPLACE "/" "_" stem_rst ${stem})
      set(docgen_target_name libc-docgen-${stem_rst})

      # docgen invocation.
      add_custom_target(${docgen_target_name}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../utils/docgen/docgen.py ${stem}.h >
                ${CMAKE_CURRENT_BINARY_DIR}/headers/${stem}.rst)
      add_dependencies(${docgen_target_name} copy-libc-rst-docs)
      # depend on the docgen invocation.
      add_dependencies(docs-libc-html ${docgen_target_name})
    endforeach()
  endif()
endif()
endif()
