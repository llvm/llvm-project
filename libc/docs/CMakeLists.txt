if(LLVM_ENABLE_SPHINX)
  include(AddSphinxTarget)
  if(SPHINX_FOUND)
    if(${SPHINX_OUTPUT_HTML})
      # Similar to clang, we copy our static documentation files from libc/docs/
      # to the $build_dir/libc/docs/. That way, we can have a mix of both static
      # committed files and dynamically generated .md files. We don't want the
      # dynamically generated .md files to pollute the source tree.
      # Maintain a list of headers for which we dynamically generate html docs
      # for via docgen. For more complex docs (such as per arch support, a la
      # math.h), those should be omitted and exist statically in
      # libc/docs/headers/.
      list(APPEND docgen_list
        aio
        arpa/inet
        assert
        byteswap
        complex
        cpio
        ctype
        dirent
        dlfcn
        endian
        errno
        fcntl
        fenv
        float
        fmtmsg
        fnmatch
        ftw
        glob
        grp
        iconv
        inttypes
        langinfo
        libgen
        limits
        locale
        math
        monetary
        mqueue
        ndbm
        netdb
        nl_types
        net/if
        netinet/in
        netinet/tcp
        poll
        pthread
        pwd
        regex
        sched
        search
        semaphore
        setjmp
        signal
        spawn
        stdbit
        stdckdint
        stdfix
        stdint
        stdio
        stdlib
        string
        strings
        sys/ipc
        sys/mman
        sys/msg
        sys/resource
        sys/select
        sys/sem
        sys/shm
        sys/socket
        sys/stat
        sys/statvfs
        sys/time
        sys/times
        sys/types
        sys/uio
        sys/un
        sys/utsname
        sys/wait
        syslog
        tar
        termios
        threads
        time
        uchar
        ulimit
        unistd
        utime
        utmpx
        wchar
        wctype
        wordexp
      )
      foreach(stem IN LISTS docgen_list)
        list(APPEND libc_generated_docs "headers/${stem}.md")
      endforeach()
      list(APPEND libc_generated_docs headers/index.md configure.rst)
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
        string(REPLACE "/" "_" stem_target ${stem})
        set(docgen_target_name libc-docgen-${stem_target})

        # docgen invocation.
        add_custom_target(${docgen_target_name}
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../utils/docgen/docgen.py ${stem}.h >
            ${CMAKE_CURRENT_BINARY_DIR}/headers/${stem}.md)
        add_dependencies(${docgen_target_name} copy-libc-rst-docs)
        # depend on the docgen invocation.
        add_dependencies(docs-libc-html ${docgen_target_name})
      endforeach()
      add_custom_target(libc-docgen-index
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../utils/docgen/docgen.py --index >
          ${CMAKE_CURRENT_BINARY_DIR}/headers/index.md)
      add_dependencies(libc-docgen-index copy-libc-rst-docs)
      add_dependencies(docs-libc-html libc-docgen-index)
    endif()
  endif()
endif()
