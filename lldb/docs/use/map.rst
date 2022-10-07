GDB to LLDB command map
=======================

Below is a table of GDB commands with the LLDB counterparts. The built in
GDB-compatibility aliases in LLDB are also listed. The full lldb command names
are often long, but any unique short form can be used. Instead of "**breakpoint
set**", "**br se**" is also acceptable.

.. contents::
   :local:

Execution Commands
------------------

.. raw:: html

   <table class="mapping" cellspacing="0">
      <tbody>
         <tr>
               <td class="hed" width="50%">GDB</td>
               <td class="hed" width="50%">LLDB</td>
         </tr>

         <tr>
               <td class="header" colspan="2">Launch a process no arguments.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> run
                  <br>
                  <b>(gdb)</b> r
               </td>
               <td class="content">
                  <b>(lldb)</b> process launch
                  <br>
                  <b>(lldb)</b> run
                  <br>
                  <b>(lldb)</b> r
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Launch a process with arguments <code>&lt;args&gt;</code>.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> run &lt;args&gt;
                  <br>
                  <b>(gdb)</b> r &lt;args&gt;
               </td>
               <td class="content">
                  <b>(lldb)</b> process launch -- &lt;args&gt;
                  <br>
                  <b>(lldb)</b> run &lt;args&gt;
                  <br>
                  <b>(lldb)</b> r &lt;args&gt;
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Launch a process for with arguments <b><code>a.out 1 2 3</code></b> without having to supply the args every time.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>%</b> gdb --args a.out 1 2 3
                  <br>
                  <b>(gdb)</b> run
                  <br> ...
                  <br>
                  <b>(gdb)</b> run
                  <br> ...
                  <br>
               </td>
               <td class="content">
                  <b>%</b> lldb -- a.out 1 2 3
                  <br>
                  <b>(lldb)</b> run
                  <br> ...
                  <br>
                  <b>(lldb)</b> run
                  <br> ...
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Or:</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> set args 1 2 3
                  <br>
                  <b>(gdb)</b> run
                  <br> ...
                  <br>
                  <b>(gdb)</b> run
                  <br> ...
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> settings set target.run-args 1 2 3
                  <br>
                  <b>(lldb)</b> run
                  <br> ...
                  <br>
                  <b>(lldb)</b> run
                  <br> ...
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Launch a process with arguments in new terminal window (macOS only).</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> process launch --tty -- &lt;args&gt;
                  <br>
                  <b>(lldb)</b> pro la -t -- &lt;args&gt;
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Launch a process with arguments in existing terminal
                  <cope>/dev/ttys006 (macOS only).</cope>
               </td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> process launch --tty=/dev/ttys006 -- &lt;args&gt;
                  <br>
                  <b>(lldb)</b> pro la -t/dev/ttys006 -- &lt;args&gt;
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set environment variables for process before launching.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> set env DEBUG 1
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> settings set target.env-vars DEBUG=1
                  <br>
                  <b>(lldb)</b> set se target.env-vars DEBUG=1
                  <br>
                  <b>(lldb)</b> env DEBUG=1
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Unset environment variables for process before launching.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> unset env DEBUG
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> settings remove target.env-vars DEBUG
                  <br>
                  <b>(lldb)</b> set rem target.env-vars DEBUG
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the arguments that will be or were passed to the program when run.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> show args
                  <br> Argument list to give program being debugged when it is started is "1 2 3".
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> settings show target.run-args
                  <br> target.run-args (array of strings) =
                  <br> [0]: "1"
                  <br> [1]: "2"
                  <br> [2]: "3"
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set environment variables for process and launch process in one command.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> process launch -E DEBUG=1
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Attach to a process with process ID 123.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> attach 123
               </td>
               <td class="content">
                  <b>(lldb)</b> process attach --pid 123
                  <br>
                  <b>(lldb)</b> attach -p 123
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Attach to a process named "a.out".</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> attach a.out
               </td>
               <td class="content">
                  <b>(lldb)</b> process attach --name a.out
                  <br>
                  <b>(lldb)</b> pro at -n a.out
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Wait for a process named "a.out" to launch and attach.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> attach -waitfor a.out
               </td>
               <td class="content">
                  <b>(lldb)</b> process attach --name a.out --waitfor
                  <br>
                  <b>(lldb)</b> pro at -n a.out -w
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Attach to a remote gdb protocol server running on system "eorgadd", port 8000.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> target remote eorgadd:8000
               </td>
               <td class="content">
                  <b>(lldb)</b> gdb-remote eorgadd:8000
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Attach to a remote gdb protocol server running on the local system, port 8000.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> target remote localhost:8000
               </td>
               <td class="content">
                  <b>(lldb)</b> gdb-remote 8000
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Attach to a Darwin kernel in kdp mode on system "eorgadd".</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> kdp-reattach eorgadd
               </td>
               <td class="content">
                  <b>(lldb)</b> kdp-remote eorgadd
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Do a source level single step in the currently selected thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> step
                  <br>
                  <b>(gdb)</b> s
               </td>
               <td class="content">
                  <b>(lldb)</b> thread step-in
                  <br>
                  <b>(lldb)</b> step
                  <br>
                  <b>(lldb)</b> s
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Do a source level single step over in the currently selected thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> next
                  <br>
                  <b>(gdb)</b> n
               </td>
               <td class="content">
                  <b>(lldb)</b> thread step-over
                  <br>
                  <b>(lldb)</b> next
                  <br>
                  <b>(lldb)</b> n
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Do an instruction level single step in the currently selected thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> stepi
                  <br>
                  <b>(gdb)</b> si
               </td>
               <td class="content">
                  <b>(lldb)</b> thread step-inst
                  <br>
                  <b>(lldb)</b> si
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Do an instruction level single step over in the currently selected thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> nexti
                  <br>
                  <b>(gdb)</b> ni
               </td>
               <td class="content">
                  <b>(lldb)</b> thread step-inst-over
                  <br>
                  <b>(lldb)</b> ni
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Step out of the currently selected frame.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> finish
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> thread step-out
                  <br>
                  <b>(lldb)</b> finish
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Return immediately from the currently selected frame, with an optional return value.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> return &lt;RETURN EXPRESSION&gt;
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> thread return &lt;RETURN EXPRESSION&gt;
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Backtrace and disassemble every time you stop.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> target stop-hook add
                  <br> Enter your stop hook command(s). Type 'DONE' to end.
                  <br> &gt; bt
                  <br> &gt; disassemble --pc
                  <br> &gt; DONE
                  <br> Stop hook #1 added.
                  <br>
               </td>
         </tr>
         <tr>
               <td class="header" colspan="2">Run until we hit line <b>12</b> or control leaves the current function.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> until 12
               </td>
               <td class="content">
                  <b>(lldb)</b> thread until 12
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the current frame and source line.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> frame
               </td>
               <td class="content">
                  <b>(lldb)</b> frame select
                  <b>(lldb)</b> f
                  <b>(lldb)</b> process status
               </td>
         </tr>

      </tbody>
   </table>


Breakpoint Commands
-------------------

.. raw:: html

   <table class="mapping" cellspacing="0">
      <tbody>
         <tr>
               <td class="hed" width="50%">GDB</td>
               <td class="hed" width="50%">LLDB</td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a breakpoint at all functions named <b>main</b>.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> break main
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint set --name main
                  <br>
                  <b>(lldb)</b> br s -n main
                  <br>
                  <b>(lldb)</b> b main
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a breakpoint in file <b>test.c</b> at line <b>12</b>.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> break test.c:12
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint set --file test.c --line 12
                  <br>
                  <b>(lldb)</b> br s -f test.c -l 12
                  <br>
                  <b>(lldb)</b> b test.c:12
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a breakpoint at all C++ methods whose basename is <b>main</b>.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> break main
                  <br>
                  <i>(Hope that there are no C functions named <b>main</b>)</i>.
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint set --method main
                  <br>
                  <b>(lldb)</b> br s -M main
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a breakpoint at and object C function: <b>-[NSString stringWithFormat:]</b>.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> break -[NSString stringWithFormat:]
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint set --name "-[NSString stringWithFormat:]"
                  <br>
                  <b>(lldb)</b> b -[NSString stringWithFormat:]
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a breakpoint at all Objective-C methods whose selector is <b>count</b>.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> break count
                  <br>
                  <i>(Hope that there are no C or C++ functions named <b>count</b>)</i>.
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint set --selector count
                  <br>
                  <b>(lldb)</b> br s -S count
                  <br>
               </td>
         </tr>
         <tr>
               <td class="header" colspan="2">Set a breakpoint by regular expression on function name.</td>
         </tr>

         <tr>
               <td class="content">
                  <b>(gdb)</b> rbreak regular-expression
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint set --func-regex regular-expression
                  <br>
                  <b>(lldb)</b> br s -r regular-expression
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Ensure that breakpoints by file and line work for #included .c/.cpp/.m files.</td>
         </tr>

         <tr>
               <td class="content">
                  <b>(gdb)</b> b foo.c:12
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> settings set target.inline-breakpoint-strategy always
                  <br>
                  <b>(lldb)</b> br s -f foo.c -l 12
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a breakpoint by regular expression on source file contents.</td>
         </tr>

         <tr>
               <td class="content">
                  <b>(gdb)</b> shell grep -e -n pattern source-file
                  <br>
                  <b>(gdb)</b> break source-file:CopyLineNumbers
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint set --source-pattern regular-expression --file SourceFile
                  <br>
                  <b>(lldb)</b> br s -p regular-expression -f file
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a conditional breakpoint</td>
         </tr>

         <tr>
               <td class="content">
                  <b>(gdb)</b> break foo if strcmp(y,"hello") == 0
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint set --name foo --condition '(int)strcmp(y,"hello") == 0'
                  <br>
                  <b>(lldb)</b> br s -n foo -c '(int)strcmp(y,"hello") == 0'
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">List all breakpoints.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info break
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint list
                  <br>
                  <b>(lldb)</b> br l
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Delete a breakpoint.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> delete 1
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint delete 1
                  <br>
                  <b>(lldb)</b> br del 1
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Disable a breakpoint.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> disable 1
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint disable 1
                  <br>
                  <b>(lldb)</b> br dis 1
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Enable a breakpoint.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> enable 1
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> breakpoint enable 1
                  <br>
                  <b>(lldb)</b> br en 1
                  <br>
               </td>
         </tr>

      </tbody>
   </table>


Watchpoint Commands
-------------------

.. raw:: html

   <table class="mapping" cellspacing="0">
      <tbody>
         <tr>
               <td class="hed" width="50%">GDB</td>
               <td class="hed" width="50%">LLDB</td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a watchpoint on a variable when it is written to.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> watch global_var
               </td>
               <td class="content">
                  <b>(lldb)</b> watchpoint set variable global_var
                  <br>
                  <b>(lldb)</b> wa s v global_var
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a watchpoint on a memory location when it is written into. The size of the region to watch for defaults to the pointer size if no '-x byte_size' is specified. This command takes raw input, evaluated as an expression returning an unsigned integer pointing to the start of the region, after the '--' option terminator.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> watch -location g_char_ptr
               </td>
               <td class="content">
                  <b>(lldb)</b> watchpoint set expression -- my_ptr
                  <br>
                  <b>(lldb)</b> wa s e -- my_ptr
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Set a condition on a watchpoint.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> watch set var global
                  <br>
                  <b>(lldb)</b> watchpoint modify -c '(global==5)'
                  <br>
                  <b>(lldb)</b> c
                  <br> ...
                  <br>
                  <b>(lldb)</b> bt
                  <br> * thread #1: tid = 0x1c03, 0x0000000100000ef5 a.out`modify + 21 at main.cpp:16, stop reason = watchpoint 1
                  <br> frame #0: 0x0000000100000ef5 a.out`modify + 21 at main.cpp:16
                  <br> frame #1: 0x0000000100000eac a.out`main + 108 at main.cpp:25
                  <br> frame #2: 0x00007fff8ac9c7e1 libdyld.dylib`start + 1
                  <br>
                  <b>(lldb)</b> frame var global
                  <br> (int32_t) global = 5
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">List all watchpoints.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info break
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> watchpoint list
                  <br>
                  <b>(lldb)</b> watch l
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Delete a watchpoint.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> delete 1
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> watchpoint delete 1
                  <br>
                  <b>(lldb)</b> watch del 1
                  <br>
               </td>
         </tr>

      </tbody>
   </table>


Examining Variables
-------------------

.. raw:: html

   <table class="mapping" cellspacing="0">
      <tbody>
         <tr>
               <td class="hed" width="50%">GDB</td>
               <td class="hed" width="50%">LLDB</td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the arguments and local variables for the current frame.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info args
                  <br> and
                  <br>
                  <b>(gdb)</b> info locals
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> frame variable
                  <br>
                  <b>(lldb)</b> fr v
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the local variables for the current frame.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info locals
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> frame variable --no-args
                  <br>
                  <b>(lldb)</b> fr v -a
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the contents of local variable "bar".</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> p bar
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> frame variable bar
                  <br>
                  <b>(lldb)</b> fr v bar
                  <br>
                  <b>(lldb)</b> p bar
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the contents of local variable "bar" formatted as hex.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> p/x bar
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> frame variable --format x bar
                  <br>
                  <b>(lldb)</b> fr v -f x bar
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the contents of global variable "baz".</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> p baz
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> target variable baz
                  <br>
                  <b>(lldb)</b> ta v baz
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the global/static variables defined in the current source file.</td>
         </tr>
         <tr>
               <td class="content">
                  n/a
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> target variable
                  <br>
                  <b>(lldb)</b> ta v
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Display the variables "argc" and "argv" every time you stop.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> display argc
                  <br>
                  <b>(gdb)</b> display argv
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> target stop-hook add --one-liner "frame variable argc argv"
                  <br>
                  <b>(lldb)</b> ta st a -o "fr v argc argv"
                  <br>
                  <b>(lldb)</b> display argc
                  <br>
                  <b>(lldb)</b> display argv
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Display the variables "argc" and "argv" only when you stop in the function named <b>main</b>.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> target stop-hook add --name main --one-liner "frame variable argc argv"
                  <br>
                  <b>(lldb)</b> ta st a -n main -o "fr v argc argv"
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Display the variable "*this" only when you stop in c class named <b>MyClass</b>.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> target stop-hook add --classname MyClass --one-liner "frame variable *this"
                  <br>
                  <b>(lldb)</b> ta st a -c MyClass -o "fr v *this"
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Print an array of integers in memory, assuming we have a pointer like "int *ptr".</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> p *ptr@10
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> parray 10 ptr
                  <br>
               </td>
         </tr>

      </tbody>
   </table>

Evaluating Expressions
----------------------

.. raw:: html

   <table class="mapping" cellspacing="0">
      <tbody>
         <tr>
               <td class="hed" width="50%">GDB</td>
               <td class="hed" width="50%">LLDB</td>
         </tr>

         <tr>
               <td class="header" colspan="2">Evaluating a generalized expression in the current frame.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> print (int) printf ("Print nine: %d.", 4 + 5)
                  <br> or if you don't want to see void returns:
                  <br>
                  <b>(gdb)</b> call (int) printf ("Print nine: %d.", 4 + 5)
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> expr (int) printf ("Print nine: %d.", 4 + 5)
                  <br> or using the print alias:
                  <br>
                  <b>(lldb)</b> print (int) printf ("Print nine: %d.", 4 + 5)
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Creating and assigning a value to a convenience variable.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> set $foo = 5
                  <br>
                  <b>(gdb)</b> set variable $foo = 5
                  <br> or using the print command
                  <br>
                  <b>(gdb)</b> print $foo = 5
                  <br> or using the call command
                  <br>
                  <b>(gdb)</b> call $foo = 5
                  <br> and if you want to specify the type of the variable:
                  <b>(gdb)</b> set $foo = (unsigned int) 5
                  <br>

               </td>
               <td class="content">
                  In lldb you evaluate a variable declaration expression as you would write it in C:
                  <br>
                  <b>(lldb)</b> expr unsigned int $foo = 5
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Printing the ObjC "description" of an object.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> po [SomeClass returnAnObject]
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> expr -o -- [SomeClass returnAnObject]
                  <br> or using the po alias:
                  <br>
                  <b>(lldb)</b> po [SomeClass returnAnObject]
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Print the dynamic type of the result of an expression.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> set print object 1
                  <br>
                  <b>(gdb)</b> p someCPPObjectPtrOrReference
                  <br> only works for C++ objects.
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> expr -d 1 -- [SomeClass returnAnObject]
                  <br>
                  <b>(lldb)</b> expr -d 1 -- someCPPObjectPtrOrReference
                  <br> or set dynamic type printing to be the default:
                  <b>(lldb)</b> settings set target.prefer-dynamic run-target
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Calling a function so you can stop at a breakpoint in the function.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> set unwindonsignal 0
                  <br>
                  <b>(gdb)</b> p function_with_a_breakpoint()
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> expr -i 0 -- function_with_a_breakpoint()
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Calling a function that crashes, and stopping when the function crashes.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> set unwindonsignal 0
                  <br>
                  <b>(gdb)</b> p function_which_crashes()
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> expr -u 0 -- function_which_crashes()
                  <br>
               </td>
         </tr>

      </tbody>
   </table>

Examining Thread State
----------------------

.. raw:: html

   <table class="mapping" cellspacing="0">
      <tbody>
         <tr>
               <td class="hed" width="50%">GDB</td>
               <td class="hed" width="50%">LLDB</td>
         </tr>

         <tr>
               <td class="header" colspan="2">List the threads in your program.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info threads
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> thread list
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Select thread 1 as the default thread for subsequent commands.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> thread 1
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> thread select 1
                  <br>
                  <b>(lldb)</b> t 1
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the stack backtrace for the current thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> bt
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> thread backtrace
                  <br>
                  <b>(lldb)</b> bt
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the stack backtraces for all threads.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> thread apply all bt
               </td>
               <td class="content">
                  <b>(lldb)</b> thread backtrace all
                  <br>
                  <b>(lldb)</b> bt all
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Backtrace the first five frames of the current thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> bt 5
               </td>
               <td class="content">
                  <b>(lldb)</b> thread backtrace -c 5
                  <br>
                  <b>(lldb)</b> bt 5 (<i>lldb-169 and later</i>)
                  <br>
                  <b>(lldb)</b> bt -c 5 (<i>lldb-168 and earlier</i>)
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Select a different stack frame by index for the current thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> frame 12
               </td>
               <td class="content">
                  <b>(lldb)</b> frame select 12
                  <br>
                  <b>(lldb)</b> fr s 12
                  <br>
                  <b>(lldb)</b> f 12
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">List information about the currently selected frame in the current thread.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> frame info
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Select the stack frame that called the current stack frame.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> up
               </td>
               <td class="content">
                  <b>(lldb)</b> up
                  <br>
                  <b>(lldb)</b> frame select --relative=1
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Select the stack frame that is called by the current stack frame.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> down
               </td>
               <td class="content">
                  <b>(lldb)</b> down
                  <br>
                  <b>(lldb)</b> frame select --relative=-1
                  <br>
                  <b>(lldb)</b> fr s -r-1
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Select a different stack frame using a relative offset.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> up 2
                  <br>
                  <b>(gdb)</b> down 3
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> frame select --relative 2
                  <br>
                  <b>(lldb)</b> fr s -r2
                  <br>
                  <br>
                  <b>(lldb)</b> frame select --relative -3
                  <br>
                  <b>(lldb)</b> fr s -r-3
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the general purpose registers for the current thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info registers
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> register read
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Write a new decimal value '123' to the current thread register 'rax'.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> p $rax = 123
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> register write rax 123
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Skip 8 bytes ahead of the current program counter (instruction pointer). Note that we use backticks to evaluate an expression and insert the scalar result in LLDB.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> jump *$pc+8
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> register write pc `$pc+8`
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the general purpose registers for the current thread formatted as <b>signed decimal</b>. LLDB tries to use the same format characters as <b>printf(3)</b> when possible. Type "help format" to see the full list of format specifiers.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> register read --format i
                  <br>
                  <b>(lldb)</b> re r -f i
                  <br>
                  <br>
                  <i>LLDB now supports the GDB shorthand format syntax but there can't be space after the command:</i>
                  <br>
                  <b>(lldb)</b> register read/d
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show all registers in all register sets for the current thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info all-registers
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> register read --all
                  <br>
                  <b>(lldb)</b> re r -a
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the values for the registers named "rax", "rsp" and "rbp" in the current thread.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info all-registers rax rsp rbp
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> register read rax rsp rbp
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show the values for the register named "rax" in the current thread formatted as <b>binary</b>.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> p/t $rax
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> register read --format binary rax
                  <br>
                  <b>(lldb)</b> re r -f b rax
                  <br>
                  <br>
                  <i>LLDB now supports the GDB shorthand format syntax but there can't be space after the command:</i>
                  <br>
                  <b>(lldb)</b> register read/t rax
                  <br>
                  <b>(lldb)</b> p/t $rax
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Read memory from address 0xbffff3c0 and show 4 hex uint32_t values.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> x/4xw 0xbffff3c0
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> memory read --size 4 --format x --count 4 0xbffff3c0
                  <br>
                  <b>(lldb)</b> me r -s4 -fx -c4 0xbffff3c0
                  <br>
                  <b>(lldb)</b> x -s4 -fx -c4 0xbffff3c0
                  <br>
                  <br>
                  <i>LLDB now supports the GDB shorthand format syntax but there can't be space after the command:</i>
                  <br>
                  <b>(lldb)</b> memory read/4xw 0xbffff3c0
                  <br>
                  <b>(lldb)</b> x/4xw 0xbffff3c0
                  <br>
                  <b>(lldb)</b> memory read --gdb-format 4xw 0xbffff3c0
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Read memory starting at the expression "argv[0]".</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> x argv[0]
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> memory read `argv[0]`
                  <br>
                  <i><b>NOTE:</b> any command can inline a scalar expression result (as long as the target is stopped) using backticks around any expression:</i>
                  <br>
                  <b>(lldb)</b> memory read --size `sizeof(int)` `argv[0]`
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Read 512 bytes of memory from address 0xbffff3c0 and save results to a local file as <b>text</b>.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> set logging on
                  <br>
                  <b>(gdb)</b> set logging file /tmp/mem.txt
                  <br>
                  <b>(gdb)</b> x/512bx 0xbffff3c0
                  <br>
                  <b>(gdb)</b> set logging off
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> memory read --outfile /tmp/mem.txt --count 512 0xbffff3c0
                  <br>
                  <b>(lldb)</b> me r -o/tmp/mem.txt -c512 0xbffff3c0
                  <br>
                  <b>(lldb)</b> x/512bx -o/tmp/mem.txt 0xbffff3c0
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Save binary memory data starting at 0x1000 and ending at 0x2000 to a file.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> dump memory /tmp/mem.bin 0x1000 0x2000
               </td>
               <td class="content">
                  <b>(lldb)</b> memory read --outfile /tmp/mem.bin --binary 0x1000 0x2000
                  <br>
                  <b>(lldb)</b> me r -o /tmp/mem.bin -b 0x1000 0x2000
                  <br>
               </td>
         </tr>
         <tr>
               <td class="header" colspan="2">Get information about a specific heap allocation (available on macOS only).</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info malloc 0x10010d680
               </td>
               <td class="content">
                  <b>(lldb)</b> command script import lldb.macosx.heap
                  <br>
                  <b>(lldb)</b> process launch --environment MallocStackLogging=1 -- [ARGS]
                  <br>
                  <b>(lldb)</b> malloc_info --stack-history 0x10010d680
                  <br>
               </td>
         </tr>
         <tr>
               <td class="header" colspan="2">Get information about a specific heap allocation and cast the result to any dynamic type that can be deduced (available on macOS only)</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> command script import lldb.macosx.heap
                  <br>
                  <b>(lldb)</b> malloc_info --type 0x10010d680
                  <br>
               </td>
         </tr>
         <tr>
               <td class="header" colspan="2">Find all heap blocks that contain a pointer specified by an expression EXPR (available on macOS only).</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> command script import lldb.macosx.heap
                  <br>
                  <b>(lldb)</b> ptr_refs EXPR
                  <br>
               </td>
         </tr>
         <tr>
               <td class="header" colspan="2">Find all heap blocks that contain a C string anywhere in the block (available on macOS only).</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> command script import lldb.macosx.heap
                  <br>
                  <b>(lldb)</b> cstr_refs CSTRING
                  <br>
               </td>
         </tr>
         <tr>
               <td class="header" colspan="2">Disassemble the current function for the current frame.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> disassemble
               </td>
               <td class="content">
                  <b>(lldb)</b> disassemble --frame
                  <br>
                  <b>(lldb)</b> di -f
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Disassemble any functions named <b>main</b>.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> disassemble main
               </td>
               <td class="content">
                  <b>(lldb)</b> disassemble --name main
                  <br>
                  <b>(lldb)</b> di -n main
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Disassemble an address range.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> disassemble 0x1eb8 0x1ec3
               </td>
               <td class="content">
                  <b>(lldb)</b> disassemble --start-address 0x1eb8 --end-address 0x1ec3
                  <br>
                  <b>(lldb)</b> di -s 0x1eb8 -e 0x1ec3
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Disassemble 20 instructions from a given address.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> x/20i 0x1eb8
               </td>
               <td class="content">
                  <b>(lldb)</b> disassemble --start-address 0x1eb8 --count 20
                  <br>
                  <b>(lldb)</b> di -s 0x1eb8 -c 20
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Show mixed source and disassembly for the current function for the current frame.</td>
         </tr>
         <tr>
               <td class="content">
                  n/a
               </td>
               <td class="content">
                  <b>(lldb)</b> disassemble --frame --mixed
                  <br>
                  <b>(lldb)</b> di -f -m
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Disassemble the current function for the current frame and show the opcode bytes.</td>
         </tr>
         <tr>
               <td class="content">
                  n/a
               </td>
               <td class="content">
                  <b>(lldb)</b> disassemble --frame --bytes
                  <br>
                  <b>(lldb)</b> di -f -b
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Disassemble the current source line for the current frame.</td>
         </tr>
         <tr>
               <td class="content">
                  n/a
               </td>
               <td class="content">
                  <b>(lldb)</b> disassemble --line
                  <br>
                  <b>(lldb)</b> di -l
               </td>
         </tr>

      </tbody>
   </table>

Executable and Shared Library Query Commands
--------------------------------------------

.. raw:: html

   <table class="mapping" cellspacing="0">
      <tbody>
         <tr>
               <td class="hed" width="50%">GDB</td>
               <td class="hed" width="50%">LLDB</td>
         </tr>

         <tr>
               <td class="header" colspan="2">List the main executable and all dependent shared libraries.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info shared
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> image list
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Look up information for a raw address in the executable or any shared libraries.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info symbol 0x1ec4
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> image lookup --address 0x1ec4
                  <br>
                  <b>(lldb)</b> im loo -a 0x1ec4
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Look up functions matching a regular expression in a binary.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info function &lt;FUNC_REGEX&gt;
                  <br>
               </td>
               <td class="content">
                  This one finds debug symbols:
                  <br>
                  <b>(lldb)</b> image lookup -r -n &lt;FUNC_REGEX&gt;
                  <br>
                  <br> This one finds non-debug symbols:
                  <br>
                  <b>(lldb)</b> image lookup -r -s &lt;FUNC_REGEX&gt;
                  <br>
                  <br> Provide a list of binaries as arguments to limit the search.
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Find full source line information.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> info line 0x1ec4
                  <br>
               </td>
               <td class="content">
                  This one is a bit messy at present. Do:
                  <br>
                  <br>
                  <b>(lldb)</b> image lookup -v --address 0x1ec4
                  <br>
                  <br> and look for the LineEntry line, which will have the full source path and line range information.
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Look up information for an address in <b>a.out</b> only.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> image lookup --address 0x1ec4 a.out
                  <br>
                  <b>(lldb)</b> im loo -a 0x1ec4 a.out
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Look up information for for a type <code>Point</code> by name.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> ptype Point
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> image lookup --type Point
                  <br>
                  <b>(lldb)</b> im loo -t Point
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Dump all sections from the main executable and any shared libraries.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> maintenance info sections
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> image dump sections
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Dump all sections in the <b>a.out</b> module.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> image dump sections a.out
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Dump all symbols from the main executable and any shared libraries.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> image dump symtab
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Dump all symbols in <b>a.out</b> and <b>liba.so</b>.</td>
         </tr>
         <tr>
               <td class="content">
               </td>
               <td class="content">
                  <b>(lldb)</b> image dump symtab a.out liba.so
                  <br>
               </td>
         </tr>

      </tbody>
   </table>

Miscellaneous
-------------

.. raw:: html

   <table class="mapping" cellspacing="0">
      <tbody>
         <tr>
               <td class="hed" width="50%">GDB</td>
               <td class="hed" width="50%">LLDB</td>
         </tr>

         <tr>
               <td class="header" colspan="2">Search command help for a keyword.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> apropos keyword
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> apropos keyword
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Echo text to the screen.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> echo Here is some text\n
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> script print "Here is some text"
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Remap source file pathnames for the debug session. If your source files are no longer located in the same location as when the program was built --- maybe the program was built on a different computer --- you need to tell the debugger how to find the sources at their local file path instead of the build system's file path.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> set pathname-substitutions /buildbot/path /my/path
                  <br>
               </td>
               <td class="content">
                  <b>(lldb)</b> settings set target.source-map /buildbot/path /my/path
                  <br>
               </td>
         </tr>

         <tr>
               <td class="header" colspan="2">Supply a catchall directory to search for source files in.</td>
         </tr>
         <tr>
               <td class="content">
                  <b>(gdb)</b> directory /my/path
                  <br>
               </td>
               <td class="content">
                  (<i>No equivalent command - use the source-map instead.</i>)
                  <br>
               </td>
         </tr>

      </tbody>
   </table>
