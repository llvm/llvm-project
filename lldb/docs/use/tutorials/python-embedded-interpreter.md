# Embedded Python Interpreter

The embedded python interpreter can be accessed in a variety of ways from
within LLDB. The easiest way is to use the lldb command script with no
arguments at the lldb command prompt:

```python3
(lldb) script
Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.
>>> 2+3
5
>>> hex(12345)
'0x3039'
>>>
```

This drops you into the embedded python interpreter. When running under the
script command, lldb sets some convenience variables that give you quick access
to the currently selected entities that characterize the program and debugger
state. In each case, if there is no currently selected entity of the
appropriate type, the variable's IsValid method will return false. These
variables are:

| Variable | Type | Equivalent | Description |
|----------|------|------------|-------------|
| `lldb.debugger` | `lldb.SBDebugger` | `SBTarget.GetDebugger` | Contains the debugger object whose `script` command was invoked. The `lldb.SBDebugger` object owns the command interpreter and all the targets in your debug session. There will always be a Debugger in the embedded interpreter. |
| `lldb.target` | `lldb.SBTarget` | `SBDebugger.GetSelectedTarget` `SBProcess.GetTarget` | Contains the currently selected target - for instance the one made with the `file` or selected by the `target select <target-index>` command. The `lldb.SBTarget` manages one running process, and all the executable and debug files for the process. |
| `lldb.process` | `lldb.SBProcess` | `SBTarget.GetProcess` `SBThread.GetProcess` | Contains the process of the currently selected target. The `lldb.SBProcess` object manages the threads and allows access to memory for the process. |
| `lldb.thread` | `lldb.SBThread` | `SBProcess.GetSelectedThread` `SBFrame.GetThread` | Contains the currently selected thread. The `lldb.SBThread` object manages the stack frames in that thread. A thread is always selected in the command interpreter when a target stops. The `thread select <thread-index>` command can be used to change the currently selected thread. So as long as you have a stopped process, there will be some selected thread. |
| `lldb.frame` | `lldb.SBFrame` | `SBThread.GetSelectedFrame` | Contains the currently selected stack frame. The `lldb.SBFrame` object manage the stack locals and the register set for that stack. A stack frame is always selected in the command interpreter when a target stops. The `frame select <frame-index>` command can be used to change the currently selected frame. So as long as you have a stopped process, there will be some selected frame. |

While extremely convenient, these variables have a couple caveats that you
should be aware of. First of all, they hold the values of the selected objects
on entry to the embedded interpreter. They do not update as you use the LLDB
API's to change, for example, the currently selected stack frame or thread.

Moreover, they are only defined and meaningful while in the interactive Python
interpreter. There is no guarantee on their value in any other situation, hence
you should not use them when defining Python formatters, breakpoint scripts and
commands (or any other Python extension point that LLDB provides). For the
latter you'll be passed an `SBDebugger`, `SBTarget`, `SBProcess`, `SBThread` or
`SBFrame` instance and you can use the functions from the "Equivalent" column
to navigate between them.

As a rationale for such behavior, consider that lldb can run in a multithreaded
environment, and another thread might call the "script" command, changing the
value out from under you.

To get started with these objects and LLDB scripting, please note that almost
all of the lldb Python objects are able to briefly describe themselves when you
pass them to the Python print function:

```python3
(lldb) script
Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.
>>> print(lldb.debugger)
Debugger (instance: "debugger_1", id: 1)
>>> print(lldb.target)
a.out
>>> print(lldb.process)
SBProcess: pid = 58842, state = stopped, threads = 1, executable = a.out
>>> print(lldb.thread)
thread #1: tid = 0x2265ce3, 0x0000000100000334 a.out`main at t.c:2:3, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
>>> print(lldb.frame)
frame #0: 0x0000000100000334 a.out`main at t.c:2:3
```