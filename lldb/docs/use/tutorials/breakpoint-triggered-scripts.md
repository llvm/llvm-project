# Breakpoint-Triggered Scripts

One very powerful use of the lldb Python API is to have a python script run
when a breakpoint gets hit. Adding python scripts to breakpoints provides a way
to create complex breakpoint conditions and also allows for smart logging and
data gathering.

When your process hits a breakpoint to which you have attached some python
code, the code is executed as the body of a function which takes three
arguments:

```python3
def breakpoint_function_wrapper(frame, bp_loc, internal_dict):
   # Your code goes here
```

or:

```python3
def breakpoint_function_wrapper(frame, bp_loc, extra_args, internal_dict):
   # Your code goes here
```

| Argument | Type | Description |
|----------|------|-------------|
| `frame` | `lldb.SBFrame` | The current stack frame where the breakpoint got hit. The object will always be valid. This `frame` argument might *not* match the currently selected stack frame found in the `lldb` module global variable `lldb.frame`. |
| `bp_loc` | `lldb.SBBreakpointLocation` | The breakpoint location that just got hit. Breakpoints are represented by `lldb.SBBreakpoint` objects. These breakpoint objects can have one or more locations. These locations are represented by `lldb.SBBreakpointLocation` objects. |
| `extra_args` | `lldb.SBStructuredData` | **Optional** If your breakpoint callback function takes this extra parameter, then when the callback gets added to a breakpoint, its contents can parametrize this use of the callback. For instance, instead of writing a callback that stops when the caller is "Foo", you could take the function name from a field in the `extra_args`, making the callback more general. The `-k` and `-v` options to `breakpoint command add` will be passed as a Dictionary in the `extra_args` parameter, or you can provide it with the SB API's. |
| `internal_dict` | `dict` | The python session dictionary as a standard python dictionary object. |

Optionally, a Python breakpoint command can return a value. Returning `False`
tells LLDB that you do not want to stop at the breakpoint. Any other return
value (including None or leaving out the return statement altogether) is akin
to telling LLDB to actually stop at the breakpoint. This can be useful in
situations where a breakpoint only needs to stop the process when certain
conditions are met, and you do not want to inspect the program state manually
at every stop and then continue.

An example will show how simple it is to write some python code and attach it
to a breakpoint. The following example will allow you to track the order in
which the functions in a given shared library are first executed during one run
of your program. This is a simple method to gather an order file which can be
used to optimize function placement within a binary for execution locality.

We do this by setting a regular expression breakpoint that will match every
function in the shared library. The regular expression '.' will match any
string that has at least one character in it, so we will use that. This will
result in one lldb.SBBreakpoint object that contains an
lldb.SBBreakpointLocation object for each function. As the breakpoint gets hit,
we use a counter to track the order in which the function at this particular
breakpoint location got hit. Since our code is passed the location that was
hit, we can get the name of the function from the location, disable the
location so we won't count this function again; then log some info and continue
the process.

Note we also have to initialize our counter, which we do with the simple
one-line version of the script command.

Here is the code:

```python3
(lldb) breakpoint set --func-regex=. --shlib=libfoo.dylib
Breakpoint created: 1: regex = '.', module = libfoo.dylib, locations = 223
(lldb) script counter = 0
(lldb) breakpoint command add --script-type python 1
Enter your Python command(s). Type 'DONE' to end.
> # Increment our counter.  Since we are in a function, this must be a global python variable
> global counter
> counter += 1
> # Get the name of the function
> name = frame.GetFunctionName()
> # Print the order and the function name
> print('[%i] %s' % (counter, name))
> # Disable the current breakpoint location so it doesn't get hit again
> bp_loc.SetEnabled(False)
> # No need to stop here
> return False
> DONE
```

The breakpoint command add command above attaches a python script to breakpoint 1. To remove the breakpoint command:

```python3
(lldb) breakpoint command delete 1
```