# Dexter Script Testing

Dexter's script mode is the default mode of operation; the heuristic mode can be used instead by using the `--use-heuristic` flag (see the main Dexter [Readme](./README.md)).

Dexter scripts are represented by YAML documents, which contain various "nodes" instructing Dexter how to step through the debuggee program, what information to collect and store from the debugger, and how to evaluate the result. A simple Dexter script looks something like this:

```yaml
---
!where {function: foo}:
    !value arg: 5
    !type arg: int
    !and {lines: !range [10, 14]}:
        !value local: ['a', 'b', 'c']
!where {function: bar}:
    !where {function: baz}:
        !step exactly: [20, 21, 22, 23, 24]
...
```

This Dexter test checks that:
- When the debugger steps into `foo`, the type and value of `arg` is always `(int) 5`.
- While the debugger is in `foo` and the current line is between 10 and 14 (inclusive), the value of `local` is each of `'a'`, `'b'`, and `'c'` at least once.
- While the debugger is in the function `baz`, which was called directly from the function `bar`, the lines that the debugger steps through exactly the lines 20-24 in order.

The Dexter test follows a structure based on the nodes - in the example above, each line starts with a node. Some of the basic types of node are:

- `!where` describes a single stack frame using either a function name or a filename + line range; these will be used by the debugger to set breakpoints. We consider a `!where` node to be "active" when the current stack frame matches the `!where` node. A `!where` node can either appear at the "root" of the script, or it can appear as the child of another `!where`, in which case it will only be active when its parent `!where` matches the frame above it. For example, the `!where {function: baz}` node is only active when the next frame up is `bar`, matching its parent. Child nodes are represented in YAML as mapping entries under the parent node, which is determined using the indent level.
- `!and` is similar to `!where`, but it can only match the same stack frame as its parent `!where` (and cannot appear at the root of the script). For example, the `!and {lines: !range [10, 14]}` node is only active when the current line number is in the range [10-14] *and* the current function is `foo`, because the `!and` is a child of `!where {function: foo}`. `!where` and `!and` nodes are collectively referred to as "state" nodes.
- `!value` and `!type` are "expect" nodes, meaning they describe testable output from the debugger. These nodes must appear a children of a state node (`!where` or `!and`), and are active whenever their parent is active. The form these nodes take is `!(value|type) <variable-name>: <expected-values>`, and their function is to collect information for `<variable-name>` while the debugger is running and the node is active, and compare that to `<expected-values>` during the evaluation step to produce the final test results.
- `!step` is another kind of expect node, which tests the line numbers seen while stepping through the program, and its expected value is a list of line numbers that we expect to see (or not see in some cases - see more detailed documentation below).
- `!range` isn't a "script node" as the others above are, but a "utility node", meaning it is used by other nodes to represent some data. `!range [<start>, <stop>]` represents an inclusive range from `start` to `stop`, and is used by state nodes.

All these nodes are arranged in a nested map structure, where each state node maps to its children. A YAML document containing this structure is embedded in the input test file: the file may either be a YAML file, where the whole file is a single document, or else the first valid YAML document contained in the file which is also a valid Dexter test script will be used. Generally, this requires one line that is just `---` to start the document, and another which is just `...` to end the document.

# Script Nodes

## State Nodes

State nodes are matched against stack frames when the debugger is stopped, and are used to control how Dexter controls the debugger (e.g. what step/continue actions to take after each stop) and determine the scope where other nodes are evaluated. State nodes have child nodes, declare some form of state that can be compared against a particular stack frame to produce either a match or non-match; when they match against the current frame, their children are evaluated. State nodes have the following rules:

- **Root** state nodes are top-level nodes in the script. Each time the debugger stops, Dexter attempts to match each root node to each stack frame, searching from the *root-most* to the *leaf-most* stack frame, and stopping at the first matching frame (if any). Root nodes must always be `!where` nodes.
- **Nested** state nodes refer to any non-root state nodes. There are two kinds of nested state node possible: `!where` and `!and`. A nested `!where` node can only match the stack frame called from the frame that matches its parent state node. A nested `!and` node can only match the same stack frame as its parent state node.
- Each state node can match only one frame per-step, e.g. if you have a recursive function `fib` and a root node `!where {function: fib}`, the node will only match the first/rootmost call, not any of the recursive calls. Conversely however, it is possible for a single frame to be matched by different state nodes.
- Both `!where` and `!and` nodes have the format: `!<type> { <args>, ... }`, supporting the following arguments:
    - `function: <function-name>` - Declares the name of a function to match on the current frame. This must be an exact match according to the debugger's presented function name, which may including namespace qualifiers. Mutually exclusive with `lines` or `file`.
    - `lines: <line> | <range>` - Declares one or more line numbers that the frame should match. Mutually exclusive with `function`; may be passed along with `file`, and if `file` is omitted for a `!where` it defaults to the script filepath, while for an `!and` node no explicit `file` is assumed, i.e. the `!and` will only match the line number and not the current file. This argument takes either a single line number, or an inclusive range of line numbers in the form `!range [<start>, <end>]`. Labels may be provided instead of literal numbers (see below).
    - `file: <file-name>` - Declares the file that the frame's source location should match. Mutually exclusive with `function`, can only be passed if `line` is also passed. If `--debugger-use-relative-paths` is passed to Dexter, then the filepath will be treated as a relative path from `--source-root-dir` when setting breakpoints; otherwise, the filepath will be passed to the debugger verbatim when setting breakpoints (which generally works for local builds).
    - `for_hit_count: <count>` - Means that the state node can only become active `count` times. A state node only "becomes" active when it was previously inactive, meaning we don't increment the hit count for a state node if was also active in the previous step.
    - `after_hit_count: <count>` - Means that the state node can only become active after it is reached `count` times. As with `for_hit_count`, the hit count is not incremented if the state node would have also been active in the previous step.
    - `conditions: <cond>` - Means that the state node is only active when the condition given by `cond` is true, which will be evaluated every step that the state node would be active. If a state node with a condition also has a child `!where` node, `cond` will *not* be re-evaluated while in the called frame - the condition is assumed to remain true until we return to the frame that contains it. The `condition` is checked before determining hit counts, i.e. the hit count for a node with `for_hit_count` or `after_hit_count` will only be incremented if `cond` is true.
- Additionally, `!and` supports one more argument:
    - `at_frame_idx: <frame-index>` - Means that instead of the `!and` node matching the same frame as its parent state node, it matches the frame at index `frame-index`, where the leaf frame has index 0. All other conditions of the `!and` and all of its children will be evaluated against that frame, providing a way to explicitly test values in frames other than the current frame. An `!and` node with `at_frame_idx` cannot contain any nested `!where` nodes.

As a convenience, Dexter supports use of "labels" instead of literal line numbers. These labels have the form `!label <name>`, and will be substituted by a line number that the label represents. Line numbers for each label come from the program source files: when Dexter encounters a `!label`, it will search the source file where that line should appear. When it encounters the string `!dex_label <name>` - regardless of surrounding context - it maps that label name to the line on which that string appears within the file. The source file that Dexter searches for labels is determined depending on the state node: for a `!where` node, either its explicit `file` argument is used, or the test file is used as the default; for an `!and` node, the label file is either the `file` argument given by the `!and` node or its nearest parent node up to the nearest `!where`, or if no explicit `file` is found then the filepath of the current frame is used as the default. Since the file should generally be a relative path, not absolute, we treat it as being relative to the `--source-root-dir` argument if it is given, or the test file's directory otherwise.

```cpp
int main() {
    for (int i = 0; i < 5; ++i)
      (void)0; // !dex_label loop
}

/*
---
!where {lines: !label loop}:
    !value i: [0, 1, 2, 3, 4]
...
*/
```

## Expect Nodes

Expect nodes describe some expected debugger output, which will be compared against the actual debugger output and be scored in Dexter's output. There are two kinds of expect node: variable expect nodes and step expect nodes.

### Variable Expect Nodes

Variable expect nodes test the debugger output for a specific variable. This takes the form of either `!type <variable>: <expected-types>` or `!value <variable>: <expected-values>`. While debugging, Dexter will fetch variable information for every variable with an active expect, and all collected values will be compared to all expected values; Dexter will report when any expected values are not seen in the output, or when any unexpected values are seen in the output. There are different forms of expected value that can be declared:

```yaml
# A single expected value may be declared for a variable, meaning Dexter expects
# the variable to have only that value while the expect is active.
!value x: 0
!type x: int

# A list of expected values means that Dexter expects the variable to have all
# of those values at least once while the expect is active; order is not
# checked, and repetitions are ignored.
!value c: ['a', 'b', 'c']
!type c: [char]

# An expected value for an aggregate, or other decomposable variable, may give
# expected values for individual members of that variable. Not all members need
# to have expected values declared.
!value tuple:
  first: 1
  second: 'z'
!type tuple:
  first: int
  second: char

# Aggregate expected values and lists of expected values can be combined: a
# variable may have a list of aggregate values, and a member expected value may
# have a list of expected values. When determining which expected value an
# aggregate matches, if there are no exact matches Dexter will select the
# closest match, and give a "partial match" score in the final output.
!value point:
  - x: 2
    y: 4
  - x: 4
    y: [8, 10]

# When testing pointers, the `!address` node may be used to specify an abstract
# label for an address and optional offset. The first time while evaluating a
# trace that Dexter matches a valid pointer against an !address, it will assume
# a valid match and assign that address to the !address label. In all future
# matches during the current evaluatoin, that !address label will be considered
# equal to the assigned address, allowing pointer variables that are always the
# same, or at a fixed offset, to be compared. 
!value iterator: [!address base, !address base + 8, !address base + 16]

# Float values can be tested using the `!float` node, which performs a float
# equality check between the expected and actual debugger output rather than a
# string comparison. An optional range may be provided which causes the `!float`
# to match any value within that range.
!value f: [!float 10, !float 5.2 +- 0.01]
```

### Step Expect Nodes

Step expect nodes test the stepping behaviour of the debugger. A step node can only have, as its expected value, a list of non-negative integers. During evaluation, Dexter will compare all of the line numbers stepped on while the step expect was active to the expected steps, according to one of 3 rules:

- `!step exactly`: while this !expect is active, we expect see exactly the expected lines in-order as many times as they appear in the expected lines list.
- `!step at_least`: while this !expect is active, we expect to see each of the expected lines in-order at least as many times as they appear in the expected list, ignoring excess lines and lines not in the expected lines list.
- `!step never`: while this !expect is active, we expect to not see any of the lines in the expected lines list. 

### Rewriting Expect Nodes

Dexter has an additional feature, "script rewriting", which can be used to generate expected values for a script from debugger output. This is used whenever an expect is declared with no expected value; this can be represented as either `!value var: null`, or `? !value var`, using the YAML syntax for a mapping key with an absent value. If at least one expect has no expected value, and a `--results-directory` is passed to Dexter, then Dexter will write a copy of the script file into the results directory; this copy will have the same name as the original file and identical contents, except that the Dexter script is replaced with a copy that has all missing expected values filled in. For example, for the following program with an embedded dexter script:

```cpp
int main() {
    for (int i = 0; i < 10; ++i)
        (void)0; // !dex_label loop
    return 0;
}

/*
---
!where {lines: !label loop}:
    ? !value i
...
*/
```

Building this program, and running Dexter using the source file as the test will result in Dexter generating a copy in the results directory that looks like:

```cpp
void foo(int);
int main() {
    for (int i = 0; i < 10; ++i)
        bar(i) // !dex_label call
    return 0;
}

/*
---
!where {lines: !label call}:
    !value i: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
...
*/
```

This works for `!value`, `!type`, and `!step` nodes, and when using script rewriting there are two additional types of expect nodes that can be used: `!value/all <scope>` and `!type/all <scope>`. These nodes fetch debugger output for *all* variables within a particular debugger scope, as defined by the DAP specification; see: https://microsoft.github.io/debug-adapter-protocol/specification#Requests_Scopes. These nodes are not directly evaluated; they must have no expected values, and when Dexter rewrites the original script, they will be replaced with `!value`/`!type` nodes for each variable that was seen in the expect's scope while it was active, inserted under !and nodes that cover that variable's live range(s). For example:

```cpp
int main() {
    int factorial = 1;
    for (int i = 1; i < 10; ++i)
        factorial *= i;
    char *string = getString(factorial);
    return 0;
}

/*
---
!where {function: main}:
    ? !value/all Locals
    ? !type/all Locals
...
*/
```

The script above, when run with Dexter using `lldb` (which defines a "Locals" scope), will produce the following script as output (this will be embedded in the output file as usual):

```yaml
!where {function: main}:
    !and {lines: !range [3, 6]}:
        !value factorial: [1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        !type factorial: int
    !and {lines: 160}:
        !value i: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        !type i: int
    !and {lines: 162}:
        !value string: "362880"
        !type string: "char*"
```

These nodes can be used to quickly generate debug info benchmark scripts for large programs by running them against an `-O0 -g` built program to capture a snapshot of what "perfect" debugging should look like in the program, which can then be compared against various build configurations, e.g. comparing optimization levels or compiler versions. These generated scripts may be less useful for comparing debug info quality across different compilers or debuggers however, since any differences in how different compilers or debuggers render variable information (e.g. different derived type rendering) will result in failed expectations.

## Execution Nodes

Execution nodes are used to directly perform debugger actions within the debugging session. The only execution node is `!then <action>`, which may appear as the sole child of a state node, e.g. `!where {lines: 10}: !then finish`. Then action requested by the `!then` node is performed after the parent state node becomes active, so in the prior example, when the debugger reaches line 10, instead of stepping off of the line as it normally would, Dexter will submit a "finish" command. There are two actions supported by `!then`:
- `!then step_out` - Performs a "step out", exiting the current function. This will disable all breakpoints for any child `!where` nodes of the currently active `!where` node, meaning this command should always succeed in exiting the current frame.
- `!then finish` - Ends the debugger session. This command allows Dexter to test programs that run for a long time, or never exit.

