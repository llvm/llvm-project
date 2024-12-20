.. include:: ../check.rst

=======
stdio.h
=======

Macros
======

.. list-table::
  :widths: auto
  :align: center
  :header-rows: 1

  * - Macro
    - Implemented
    - C23 Standard Section
    - POSIX Docs
  * - BUFSIZ
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - EOF
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - FILENAME_MAX
    -
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - FOPEN_MAX
    -
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - L_ctermid
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - L_tmpnam
    -
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - SEEK_CUR
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - SEEK_END
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - SEEK_SET
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - TMP_MAX
    -
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - _IOFBF
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - _IOLBF
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - _IONBF
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - _PRINTF_NAN_LEN_MAX
    -
    - 7.23.1
    -
  * - __STDC_VERSION_STDIO_H__
    -
    - 7.23.1
    -
  * - stderr
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - stdin
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__
  * - stdout
    - |check|
    - 7.23.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`__

Functions
=========

.. list-table::
  :widths: auto
  :align: center
  :header-rows: 1

  * - Function
    - Implemented
    - C23 Standard Section
    - POSIX Docs
  * - clearerr
    - |check|
    - 7.23.10.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/clearerr.html>`__
  * - ctermid
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/ctermid.html>`__
  * - dprintf
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/dprintf.html>`__
  * - fclose
    - |check|
    - 7.23.5.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fclose.html>`__
  * - fdopen
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fdopen.html>`__
  * - feof
    - |check|
    - 7.23.10.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/feof.html>`__
  * - ferror
    - |check|
    - 7.23.10.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/ferror.html>`__
  * - fflush
    - |check|
    - 7.23.5.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fflush.html>`__
  * - fgetc
    - |check|
    - 7.23.7.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fgetc.html>`__
  * - fgetpos
    -
    - 7.23.9.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fgetpos.html>`__
  * - fgets
    - |check|
    - 7.23.7.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fgets.html>`__
  * - fileno
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fileno.html>`__
  * - flockfile
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/flockfile.html>`__
  * - fmemopen
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fmemopen.html>`__
  * - fopen
    - |check|
    - 7.23.5.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fopen.html>`__
  * - fprintf
    - |check|
    - 7.23.6.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fprintf.html>`__
  * - fputc
    - |check|
    - 7.23.7.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fputc.html>`__
  * - fputs
    - |check|
    - 7.23.7.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fputs.html>`__
  * - fread
    - |check|
    - 7.23.8.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fread.html>`__
  * - freopen
    -
    - 7.23.5.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/freopen.html>`__
  * - fscanf
    - |check|
    - 7.23.6.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fscanf.html>`__
  * - fseek
    - |check|
    - 7.23.9.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fseek.html>`__
  * - fseeko
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fseeko.html>`__
  * - fsetpos
    -
    - 7.23.9.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fsetpos.html>`__
  * - ftell
    - |check|
    - 7.23.9.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/ftell.html>`__
  * - ftello
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/ftello.html>`__
  * - ftrylockfile
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/ftrylockfile.html>`__
  * - funlockfile
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/funlockfile.html>`__
  * - fwrite
    - |check|
    - 7.23.8.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fwrite.html>`__
  * - getc
    - |check|
    - 7.23.7.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/getc.html>`__
  * - getchar
    - |check|
    - 7.23.7.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/getchar.html>`__
  * - getdelim
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/getdelim.html>`__
  * - getline
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/getline.html>`__
  * - open_memstream
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/open_memstream.html>`__
  * - pclose
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/pclose.html>`__
  * - perror
    -
    - 7.23.10.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/perror.html>`__
  * - popen
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/popen.html>`__
  * - printf
    - |check|
    - 7.23.6.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/printf.html>`__
  * - putc
    - |check|
    - 7.23.7.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/putc.html>`__
  * - putc_unlocked
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/putc_unlocked.html>`__
  * - putchar
    - |check|
    - 7.23.7.8
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/putchar.html>`__
  * - putchar_unlocked
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/putchar_unlocked.html>`__
  * - puts
    - |check|
    - 7.23.7.9
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/puts.html>`__
  * - remove
    - |check|
    - 7.23.4.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/remove.html>`__
  * - rename
    - |check|
    - 7.23.4.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/rename.html>`__
  * - renameat
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/renameat.html>`__
  * - rewind
    -
    - 7.23.9.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/rewind.html>`__
  * - scanf
    - |check|
    - 7.23.6.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/scanf.html>`__
  * - setbuf
    - |check|
    - 7.23.5.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/setbuf.html>`__
  * - setvbuf
    - |check|
    - 7.23.5.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/setvbuf.html>`__
  * - snprintf
    - |check|
    - 7.23.6.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/snprintf.html>`__
  * - sprintf
    - |check|
    - 7.23.6.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/sprintf.html>`__
  * - sscanf
    - |check|
    - 7.23.6.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/sscanf.html>`__
  * - tmpfile
    -
    - 7.23.4.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/tmpfile.html>`__
  * - tmpnam
    -
    - 7.23.4.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/tmpnam.html>`__
  * - ungetc
    - |check|
    - 7.23.7.10
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/ungetc.html>`__
  * - vdprintf
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/vdprintf.html>`__
  * - vfprintf
    - |check|
    - 7.23.6.8
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/vfprintf.html>`__
  * - vfscanf
    - |check|
    - 7.23.6.9
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/vfscanf.html>`__
  * - vprintf
    - |check|
    - 7.23.6.10
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/vprintf.html>`__
  * - vscanf
    - |check|
    - 7.23.6.11
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/vscanf.html>`__
  * - vsnprintf
    - |check|
    - 7.23.6.12
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/vsnprintf.html>`__
  * - vsprintf
    - |check|
    - 7.23.6.13
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/vsprintf.html>`__
  * - vsscanf
    - |check|
    - 7.23.6.14
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/vsscanf.html>`__
