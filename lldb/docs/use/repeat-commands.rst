Repeat Commands
===============

In LLDB's command line interface, pressing Enter (an empty command) repeats the
previous command. By default, the exact same command is re-executed. However,
several commands customize this behavior to implement paging or progressive
expansion, making it easy to explore data incrementally by pressing Enter
repeatedly. Repeat commands can be learned and discovered by pressing enter and
observing the result.

In some instances, commands disable repeat commands, to prevent
accidentally triggering a destructive operation (e.g. ``process launch``).

This page documents the commands with custom repeat behavior.

``thread backtrace`` (``bt``)
-----------------------------

When ``thread backtrace`` is invoked with the ``--count`` (``-c``) option, the
repeat command pages through the backtrace by advancing the ``--start`` (``-s``)
option by the count each time.

For example:

::

   (lldb) bt 5
   * thread #1, stop reason = breakpoint 1.1
     * frame #0: handle_request at server.cpp:50
       frame #1: parse_headers at http.cpp:12
       frame #2: read_socket at socket.cpp:30
       frame #3: accept_connection at listener.cpp:8
       frame #4: event_loop at reactor.cpp:100
   (lldb)
   # repeats as: thread backtrace -c 5 -s 5
   * thread #1, stop reason = breakpoint 1.1
       frame #5: start_server at main.cpp:42
       frame #6: load_config at config.cpp:7
       ...
   (lldb)
   # repeats as: thread backtrace -c 5 -s 10

Each press of Enter shows the next 5 frames. If ``--count`` is not specified,
the full backtrace is displayed and there is no repeat command.

``source list`` (``list``)
--------------------------

When ``source list`` is repeated, it shows the next block of source lines,
continuing from where the previous listing ended. This mimics paging behavior.

If the ``--reverse`` (``-r``) option was given, the repeat command continues
listing in reverse (showing earlier source lines).

::

   (lldb) list          # shows source around the current location
   (lldb)               # shows the next block of source lines
   (lldb)               # shows the next block after that

   (lldb) list -r       # shows source in reverse
   (lldb)               # continues listing in reverse

``memory read``
---------------

When ``memory read`` is repeated, it continues reading from where the previous
read ended. The repeat command drops the address arguments and re-uses the same
format, size, and count options from the previous invocation.

::

   (lldb) memory read 0x1000 -c 32
   0x1000: 48 8b 05 a9 3b 00 00 48 ...
   (lldb)               # continues reading the next 32 bytes from 0x1020
   (lldb)               # continues reading from 0x1040

``memory region``
-----------------

When ``memory region`` is given an address, it displays the memory region
containing that address. Pressing Enter then shows the next memory region, and
so on, allowing you to walk through the process's entire memory map.

::

   (lldb) memory region 0x1000
   [0x0000-0x2000) rw-
   (lldb)               # shows the next region starting at 0x2000
   (lldb)               # shows the next region after that

``frame variable`` (``v``)
--------------------------

When ``frame variable`` is repeated, it re-runs the command with an incremented
``--depth`` (``-D``) value. This progressively reveals deeper levels of nested
data structures with each press of Enter.

If no ``--depth`` option was specified in the original command, the next repeat
starts at one level beyond the ``target.max-children-depth`` default setting. If
``--depth`` was specified, it increments the given value by 1 each time.

Consider a deeply nested configuration structure:

::

   (lldb) v config
   (Config) config = {
     server = {
       network = {
         tls = {
           certificate = {
             issuer = {...}
           }
         }
       }
     }
   }
   (lldb)               # repeats as: frame variable --depth 6 config
   (Config) config = {
     server = {
       network = {
         tls = {
           certificate = {
             issuer = {
               name = {...}
             }
           }
         }
       }
     }
   }
   (lldb)               # repeats as: frame variable --depth 7 config

The default ``target.max-children-depth`` causes the first output to truncate at
``issuer``. Each press of Enter reveals one more level without having to
manually specify ``--depth``.

``thread trace dump instructions``
----------------------------------

When repeated, this command adds the ``--continue`` flag, which continues
dumping traced instructions from where the previous instruction dump left off.

::

   (lldb) thread trace dump instructions
   ... first batch of instructions ...
   (lldb)               # continues dumping the next 20 instructions
