# Dexter commands

* [DexExpectProgramState](Commands.md#DexExpectProgramState)
* [DexExpectStepKind](Commands.md#DexExpectStepKind)
* [DexExpectStepOrder](Commands.md#DexExpectStepOrder)
* [DexExpectWatchType](Commands.md#DexExpectWatchType)
* [DexExpectWatchValue](Commands.md#DexExpectWatchValue)
* [DexUnreachable](Commands.md#DexUnreachable)
* [DexWatch](Commands.md#DexWatch)

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
    DexExpectStepOrder(*order)

    Arg list:
      order (int): One or more indices.

### Description
Expect the line every `DexExpectStepOrder` is found on to be stepped on in
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
                        [,**on_line][,**require_in_order=True])

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

### Description
Expect the expression `expr` to evaluate to the list of `values`
sequentially.

### Heuristic
[TODO]


---
## DexUnreachable
    DexUnreachable()

### Description
Expect the source line this is found on will never be stepped on to.

### Heuristic
[TODO]


----
## DexLabel
    DexLabel(name)

    Args:
        name (str): A unique name for this line.

### Description
Name the line this command is found on. Line names can be referenced by other
commands expecting line number arguments.
For example, `DexExpectWatchValues(..., on_line='my_line_name')`.

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
