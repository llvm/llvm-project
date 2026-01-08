# Dexter commands

* [DexExpectProgramState](Commands.md#DexExpectProgramState)
* [DexExpectStepKind](Commands.md#DexExpectStepKind)
* [DexExpectStepOrder](Commands.md#DexExpectStepOrder)
* [DexExpectWatchType](Commands.md#DexExpectWatchType)
* [DexExpectWatchValue](Commands.md#DexExpectWatchValue)
* [DexUnreachable](Commands.md#DexUnreachable)
* [DexLimitSteps](Commands.md#DexLimitSteps)
* [DexLabel](Commands.md#DexLabel)
* [DexWatch](Commands.md#DexWatch)
* [DexDeclareAddress](Commands.md#DexDeclareAddress)
* [DexDeclareFile](Commands.md#DexDeclareFile)
* [DexFinishTest](Commands.md#DexFinishTest)
* [DexCommandLine](Commands.md#DexCommandLine)
* [DexStepFunction](Commands.md#DexStepFunction)
* [DexContinue](Commands.md#DexContinue)
---
## DexExpectProgramState
    DexExpectProgramState(state [,**times])

    Args:
        state (dict): { 'frames': [
                        {
                          # StackFrame #
                          'function': name (str),
                          'is_inlined': bool,
                          'location': {
                            # SourceLocation #
                            'lineno': int,
                            'path': str,
                            'column': int,
                          },
                          'watches': {
                            expr (str): value (str),
                            expr (str): {
                              'value': str,
                              'type_name': str,
                              'could_evaluate': bool,
                              'is_optimized_away': bool,
                              'is_irretrievable': bool,
			    }
                          },
                        }
                      ]}

    Keyword args:
        times (int): Minimum number of times this state pattern is expected to
             be seen. Defaults to 1. Can be 0.

### Description
Expect to see a given program `state` a certain number of `times`.

For every debugger step the reported state is compared with the expected state.
To consider the states a match:

* The `SourceLocation` must match in both states. Omitted fields in the
`SourceLocation` dictionary are ignored; they always match.
* Each `expr` in `watches` in the expected state can either be a dictionary
with the fields shown above, or a string representing its value. In either
case, the actual value of `expr` in the debugger must match.
* The function name and inline status are not considered.

### Heuristic
[TODO]


---
## DexExpectStepKind
    DexExpectStepKind(kind, times)

    Args:
      kind (str): Expected step kind.
      times (int): Expected number of encounters.

### Description
Expect to see a particular step `kind` a number of `times` while stepping
through the program.

`kind` must be one of:

`FUNC`: The first step into a function which is defined in the test
directory.</br>
`FUNC_EXTERNAL`: A step over a function which is not defined in the test
directory.</br>
`FUNC_UNKNOWN`: The first step over a function an unknown definition
location.</br>
`VERTICAL_FORWARD`: A step onto a line after the previous step line in this
frame.</br>
`VERTICAL_BACKWARD`: A step onto a line before the previous step line in
this frame.</br>
`HORIZONTAL_FORWARD`: A step forward on the same line as the previous step in
this frame.</br>
`HORIZONTAL_BACKWARD`: A step backward on the same line as the previous step
in this frame.</br>
`SAME`: A step onto the same line and column as the previous step in this
frame.</br>

### Heuristic
[TODO]


---
## DexExpectStepOrder
    DexExpectStepOrder(*order [,**on_line])

    Arg list:
      order (int): One or more indices.

    Keyword args:
        on_line (int): Expect this line to be stepped on in the order given.

### Description
Expect the line every `DexExpectStepOrder` is found on, or given from `on_line`, to be stepped on in
`order`. Each instance must have a set of unique ascending indices.

### Heuristic
[TODO]


---
## DexExpectWatchType
    DexExpectWatchType(expr, *types [,**from_line=1][,**to_line=Max]
                        [,**on_line][,**require_in_order=True])

    Args:
        expr (str): expression to evaluate.

    Arg list:
        types (str): At least one expected type. NOTE: string type.

    Keyword args:
        from_line (int): Evaluate the expression from this line. Defaults to 1.
        to_line (int): Evaluate the expression to this line. Defaults to end of
            source.
        on_line (int): Only evaluate the expression on this line. If provided,
            this overrides from_line and to_line.
        require_in_order (bool): If False the values can appear in any order.

### Description
Expect the expression `expr` to evaluate be evaluated and have each evaluation's
type checked against the list of `types`

### Heuristic
[TODO]


---
## DexExpectWatchValue
    DexExpectWatchValue(expr, *values [,**from_line=1][,**to_line=Max]
                        [,**on_line][,**require_in_order=True][,**float_range])

    Args:
        expr (str): expression to evaluate.

    Arg list:
        values (str): At least one expected value. NOTE: string type.

    Keyword args:
        from_line (int): Evaluate the expression from this line. Defaults to 1.
        to_line (int): Evaluate the expression to this line. Defaults to end of
            source.
        on_line (int): Only evaluate the expression on this line. If provided,
            this overrides from_line and to_line.
        require_in_order (bool): If False the values can appear in any order.
        float_range (float): If provided, `values` must be floats, and will
            match an actual value if they are within `float_range` of each other.


### Description
Expect the expression `expr` to evaluate to the list of `values`
sequentially.

### Heuristic
[TODO]


---
## DexUnreachable
    DexUnreachable([, **from_line=1][,**to_line=Max][,**on_line])

### Description
Expect the source line this is found on will never be stepped on to. If either
'on_line' or both 'from_line' and 'to_line' are specified, checks that the
specified line(s) are not stepped on.

### Heuristic
[TODO]


----
## DexLimitSteps
    DexLimitSteps([expr, *values][, **from_line=1][,**to_line=Max]
                  [,**on_line][,**hit_count])

    Args:
        expr (str): variable or value to compare.

    Arg list:
        values (str): At least one potential value the expr may evaluate to.

    Keyword args:
        from_line (int): Define the start of the limited step range.
        to_line (int): Define the end of the limited step range.
        on_line (int): Define a range with length 1 starting and ending on the
                       same line.
        hit_count (int): If provided, limit the number of times the command
                         triggers.

### Description
Define a limited stepping range that may be predicated on a condition. When the
leading line is stepped on and any condition '(expr) == (values[n])' is true or
there are no conditions, set a range of temporary breakpoints within the test
file defined by the range 'from_line' and 'to_line' or 'on_line'. This only
happens 'hit_count' number of times if the argument is provided.

The condition is only evaluated on the line 'from_line' or 'on_line'. If the
condition is not true at the start of the range, or that line is never stepped
onto, the whole range is ignored.

DexLimitSteps commands are useful for reducing the amount of steps gathered in
large test cases that would normally take much longer to complete.

----
## DexLabel
    DexLabel(name [, **on_line])

    Args:
        name (str): A unique name for this line.

    Keyword args:
        on_line (int): Specify a line number to label.

### Description
Name the line this command is found on or 'on_line' if it is provided. Line
names can be converted to line numbers with the `ref(str)` function. For
example, `DexExpectWatchValues(..., on_line=ref('my_line_name'))`. Use
arithmetic operators to get offsets from labels:

    DexExpectWatchValues(..., on_line=ref('my_line_name') + 3)
    DexExpectWatchValues(..., on_line=ref('my_line_name') - 5)


### Heuristic
This command does not contribute to the heuristic score.

----
## DexDeclareAddress
    DexDeclareAddress(declared_address, expr, **on_line[, **hit_count])

    Args:
        declared_address (str): The unique name of an address, which can be used
                                in DexExpectWatch-commands.
        expr (str): An expression to evaluate to provide the value of this
                    address.
        on_line (int): The line at which the value of the expression will be
                       assigned to the address.
        hit_count (int): If provided, reads the value of the source expression
                         after the line has been stepped onto the given number
                         of times ('hit_count = 0' gives default behaviour).

### Description
Declares a variable that can be used in DexExpectWatch- commands as an expected
value by using the `address(str[, int])` function. This is primarily
useful for checking the values of pointer variables, which are generally
determined at run-time (and so cannot be consistently matched by a hard-coded
expected value), but may be consistent relative to each other. An example use of
this command is as follows, using a set of pointer variables "foo", "bar", and
"baz":

    DexDeclareAddress('my_addr', 'bar', on_line=12)
    DexExpectWatchValue('foo', address('my_addr'), on_line=10)
    DexExpectWatchValue('bar', address('my_addr'), on_line=12)
    DexExpectWatchValue('baz', address('my_addr', 16), on_line=14)

On the first line, we declare the name of our variable 'my_addr'. This name must
be unique (the same name cannot be declared twice), and attempting to reference
an undeclared variable with `address` will fail. The value of the address
variable will be assigned as the value of 'bar' when line 12 is first stepped
on.

On lines 2-4, we use the `address` function to refer to our variable. The first
usage occurs on line 10, before the line where 'my_addr' is assigned its value;
this is a valid use, as we assign the address value and check for correctness
after gathering all debug information for the test. Thus the first test command
will pass if 'foo' on line 10 has the same value as 'bar' on line 12.

The second command will pass iff 'bar' is available at line 12 - even if the
variable and lines are identical in DexDeclareAddress and DexExpectWatchValue,
the latter will still expect a valid value. Similarly, if the variable for a
DexDeclareAddress command is not available at the given line, any test against
that address will fail.

The `address` function also accepts an optional integer argument representing an
offset (which may be negative) to be applied to the address value, so
`address('my_addr', 16)` resolves to `my_addr + 16`. In the above example, this
means that we expect `baz == bar + 16`.

### Heuristic
This command does not contribute to the heuristic score.

----
## DexDeclareFile
    DexDeclareFile(declared_file)

    Args:
        name (str): A declared file path for which all subsequent commands
          will have their path attribute set too.

### Description
Set the path attribute of all commands from this point in the test onwards.
The new path holds until the end of the test file or until a new DexDeclareFile
command is encountered. Used in conjunction with .dex files, DexDeclareFile can
be used to write your dexter commands in a separate test file avoiding inlined
Dexter commands mixed with test source.

### Heuristic
This command does not contribute to the heuristic score.

----
## DexFinishTest
    DexFinishTest([expr, *values], **on_line[, **hit_count=0])

    Args:
        expr (str): variable or value to compare.

    Arg list:
        values (str): At least one potential value the expr may evaluate to.

    Keyword args:
        on_line (int): Define the line on which this command will be triggered.
        hit_count (int): If provided, triggers this command only after the line
                         and condition have been hit the given number of times.

### Description
Defines a point at which Dexter will exit out of the debugger without waiting
for the program to finish. This is primarily useful for testing a program that
either does not automatically terminate or would otherwise continue for a long
time after all test commands have finished.

The command will trigger when the line 'on_line' is stepped on and either the
condition '(expr) == (values[n])' is true or there are no conditions. If the
optional argument 'hit_count' is provided, then the command will not trigger
for the first 'hit_count' times the line and condition are hit.

### Heuristic
This command does not contribute to the heuristic score.

----
## DexCommandLine
    DexCommandLine(command_line)

    Args:
        command_line (list): List of strings that form the command line.

### Description
Specifies the command line with which to launch the test. The arguments will
be appended to the default command line, i.e. the path to the compiled binary,
and will be passed to the program under test.

This command does not contribute to any part of the debug experience testing or
runtime instrumentation -- it's only for communicating arguments to the program
under test.

### Heuristic
This command does not contribute to the heuristic score.

---
## DexWatch
    DexWatch(*expressions)

    Arg list:
        expressions (str): `expression` to evaluate on this line.

### Description
[Deprecated] Evaluate each given `expression` when the debugger steps onto the
line this command is found on.

### Heuristic
[Deprecated]


---
## DexStepFunction
    DexStepFunction(function_name[, **hit_count=0])

    Arg list:
        function_name (str): function to step through.
        hit_count (int): If provided, limit the number of times the command
                         triggers.

### Description
NOTE: Only supported for DAP-based debuggers.

This command controls stepping behaviour: Tell dexter to set a function
breakpoint and step through the function after hitting it. Composes well with
itself (you can may a callstack with multiple targets to step through) and
`DexContinue`.

---
## DexContinue
    DexContinue(*[expr, *values], **from_line[, **to_line, **hit_count])

    Arg list:
        function_name (str): function to step through.
        hit_count (int): If provided, limit the number of times the command
                         triggers.

### Description
NOTE: Only supported for DAP-based debuggers.

This command controls stepping behaviour: Tell dexter to set a breakpoint on
`from_line`. When it is hit and optionally '(expr) == (values[n])' is true,
optionally set a breakpoint on `to_line`. Then 'continue' (tell the debugger to
run freely until a breakpoint is hit). Composed with `DexStepFunction` this
lets you avoid stepping over certain regions (many loop iterations, for
example). Continue-ing off the end of a `DexStepFunction` is well defined;
stepping will resume in `DexStepFunction` targets deeper in the callstack.

FIXME: hit_count should probably be inverted, like `DexLimitSteps`, to trigger
AFTER that many hits?
