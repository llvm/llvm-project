.. title:: clang-tidy - bugprone-unused-return-value

bugprone-unused-return-value
============================

Warns on unused function return values. The checked functions can be configured.

Options
-------

.. option:: CheckedFunctions

   Semicolon-separated list of functions to check. The function is checked if
   the name and scope matches, with any arguments.
   By default the following functions are checked:
   ``std::async, std::launder, std::remove, std::remove_if, std::unique,
   std::unique_ptr::release, std::basic_string::empty, std::vector::empty,
   std::back_inserter, std::distance, std::find, std::find_if, std::inserter,
   std::lower_bound, std::make_pair, std::map::count, std::map::find,
   std::map::lower_bound, std::multimap::equal_range,
   std::multimap::upper_bound, std::set::count, std::set::find, std::setfill,
   std::setprecision, std::setw, std::upper_bound, std::vector::at,
   bsearch, ferror, feof, isalnum, isalpha, isblank, iscntrl, isdigit, isgraph,
   islower, isprint, ispunct, isspace, isupper, iswalnum, iswprint, iswspace,
   isxdigit, memchr, memcmp, strcmp, strcoll, strncmp, strpbrk, strrchr,
   strspn, strstr, wcscmp, access, bind, connect, difftime, dlsym, fnmatch,
   getaddrinfo, getopt, htonl, htons, iconv_open, inet_addr, isascii, isatty,
   mmap, newlocale, openat, pathconf, pthread_equal, pthread_getspecific,
   pthread_mutex_trylock, readdir, readlink, recvmsg, regexec, scandir,
   semget, setjmp, shm_open, shmget, sigismember, strcasecmp, strsignal,
   ttyname``

   - ``std::async()``. Not using the return value makes the call synchronous.
   - ``std::launder()``. Not using the return value usually means that the
     function interface was misunderstood by the programmer. Only the returned
     pointer is "laundered", not the argument.
   - ``std::remove()``, ``std::remove_if()`` and ``std::unique()``. The returned
     iterator indicates the boundary between elements to keep and elements to be
     removed. Not using the return value means that the information about which
     elements to remove is lost.
   - ``std::unique_ptr::release()``. Not using the return value can lead to
     resource leaks if the same pointer isn't stored anywhere else. Often,
     ignoring the ``release()`` return value indicates that the programmer
     confused the function with ``reset()``.
   - ``std::basic_string::empty()`` and ``std::vector::empty()``. Not using the
     return value often indicates that the programmer confused the function with
     ``clear()``.

.. option:: CheckedReturnTypes

   Semicolon-separated list of function return types to check.
   By default the following function return types are checked:
   `::std::error_code`, `::std::error_condition`, `::std::errc`, `::std::expected`, `::boost::system::error_code`

.. option:: AllowCastToVoid

   Controls whether casting return values to ``void`` is permitted. Default: `false`.

:doc:`cert-err33-c <../cert/err33-c>` is an alias of this check that checks a
fixed and large set of standard library functions.
