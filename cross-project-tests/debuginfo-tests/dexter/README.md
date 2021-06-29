# DExTer (Debugging Experience Tester)

## Introduction

DExTer is a suite of tools used to evaluate the "User Debugging Experience". DExTer drives an external debugger, running on small test programs, and collects information on the behavior at each debugger step to provide quantitative values that indicate the quality of the debugging experience.

## Supported Debuggers

DExTer currently supports the Visual Studio 2015 and Visual Studio 2017 debuggers via the [DTE interface](https://docs.microsoft.com/en-us/dotnet/api/envdte.dte), and LLDB via its [Python interface](https://lldb.llvm.org/python-reference.html). GDB is not currently supported.

The following command evaluates your environment, listing the available and compatible debuggers:

    dexter.py list-debuggers

## Dependencies
[TODO] Add a requirements.txt or an install.py and document it here.

### Python 3.6

DExTer requires python version 3.6 or greater.

### pywin32 python package

This is required to access the DTE interface for the Visual Studio debuggers.

    <python-executable> -m pip install pywin32

### clang

DExTer is current compatible with 'clang' and 'clang-cl' compiler drivers.  The compiler must be available for DExTer, for example the following command should successfully build a runnable executable.

     <compiler-executable> tests/nostdlib/fibonacci/test.cpp

## Running a test case

The following DExTer commands build the test.cpp from the tests/nostdlib/fibonacci directory and quietly runs it on the Visual Studio debugger, reporting the debug experience heuristic.  The first command builds with no optimizations (/Od) and scores 1.0000.  The second command builds with optimizations (/Ox) and scores 0.2832 which suggests a worse debugging experience.

    dexter.py test --builder clang-cl_vs2015 --debugger vs2017 --cflags="/Od /Zi" --ldflags="/Zi" -- tests/nostdlib/fibonacci
    fibonacci = (1.0000)

    dexter.py test --builder clang-cl_vs2015 --debugger vs2017 --cflags="/Ox /Zi" --ldflags="/Zi" -- tests/nostdlib/fibonacci
    fibonacci = (0.2832)

## An example test case

The sample test case (tests/nostdlib/fibonacci) looks like this:

    1.  #ifdef _MSC_VER
    2.  # define DEX_NOINLINE __declspec(noinline)
    3.  #else
    4.  # define DEX_NOINLINE __attribute__((__noinline__))
    5.  #endif
    6.
    7.  DEX_NOINLINE
    8.  void Fibonacci(int terms, int& total)
    9.  {
    0.      int first = 0;
    11.     int second = 1;
    12.     for (int i = 0; i < terms; ++i)
    13.     {
    14.         int next = first + second; // DexLabel('start')
    15.         total += first;
    16.         first = second;
    17.         second = next;             // DexLabel('end')
    18.     }
    19. }
    20.
    21. int main()
    22. {
    23.     int total = 0;
    24.     Fibonacci(5, total);
    25.     return total;
    26. }
    27.
    28. /*
    29. DexExpectWatchValue('i', '0', '1', '2', '3', '4',
    30.                     from_line='start', to_line='end')
    31. DexExpectWatchValue('first', '0', '1', '2', '3', '5',
    32.                     from_line='start', to_line='end')
    33. DexExpectWatchValue('second', '1', '2', '3', '5',
    34                      from_line='start', to_line='end')
    35. DexExpectWatchValue('total', '0', '1', '2', '4', '7',
    36.                     from_line='start', to_line='end')
    37. DexExpectWatchValue('next', '1', '2', '3', '5', '8',
    38.                     from_line='start', to_line='end')
    39. DexExpectWatchValue('total', '7', on_line=25)
    40. DexExpectStepKind('FUNC_EXTERNAL', 0)
    41. */

[DexLabel][1] is used to give a name to a line number.

The [DexExpectWatchValue][2] command states that an expression, e.g. `i`, should
have particular values, `'0', '1', '2', '3','4'`, sequentially over the program
lifetime on particular lines. You can refer to a named line or simply the line
number (See line 39).

At the end of the test is the following line:

    DexExpectStepKind('FUNC_EXTERNAL', 0)

This [DexExpectStepKind][3] command indicates that we do not expect the debugger
to step into a file outside of the test directory.

[1]: Commands.md#DexLabel
[2]: Commands.md#DexExpectWatchValue
[3]: Commands.md#DexExpectStepKind

## Detailed DExTer reports

Running the command below launches the tests/nostdlib/fibonacci test case in DExTer, using clang-cl as the compiler, Visual Studio 2017 as the debugger, and producing a detailed report:

    $ dexter.py test --builder clang-cl_vs2015 --debugger vs2017 --cflags="/Ox /Zi" --ldflags="/Zi" -v -- tests/nostdlib/fibonacci

The detailed report is enabled by `-v` and shows a breakdown of the information from each debugger step. For example:

    fibonacci = (0.2832)

    ## BEGIN ##
    [1, "main", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 23, 1, "BREAKPOINT", "FUNC", {}]
    [2, "main", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 24, 1, "BREAKPOINT", "VERTICAL_FORWARD", {}]
    [3, "main", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 25, 1, "BREAKPOINT", "VERTICAL_FORWARD", {}]
    .   [4, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 13, 1, "BREAKPOINT", "FUNC", {}]
    .   [5, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 16, 1, "BREAKPOINT", "VERTICAL_FORWARD", {"i": "Variable is optimized away and not available.", "next": "Variable is optimized away and not available.", "second": "Variable is optimized away and not available.", "total": "0", "first": "Variable is optimized away and not available."}]
    .   [6, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 13, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {}]
    .   [7, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 15, 1, "BREAKPOINT", "VERTICAL_FORWARD", {"i": "Variable is optimized away and not available.", "second": "Variable is optimized away and not available.", "total": "0", "first": "Variable is optimized away and not available."}]
    .   [8, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 16, 1, "BREAKPOINT", "VERTICAL_FORWARD", {"i": "Variable is optimized away and not available.", "next": "Variable is optimized away and not available.", "second": "Variable is optimized away and not available.", "total": "0", "first": "Variable is optimized away and not available."}]
    .   [9, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 15, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {"i": "Variable is optimized away and not available.", "second": "1", "total": "0", "first": "0"}]
    .   [10, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 13, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {}]
    .   [11, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 16, 1, "BREAKPOINT", "VERTICAL_FORWARD", {"i": "Variable is optimized away and not available.", "next": "Variable is optimized away and not available.", "second": "Variable is optimized away and not available.", "total": "0", "first": "Variable is optimized away and not available."}]
    .   [12, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 15, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {"i": "Variable is optimized away and not available.", "second": "1", "total": "0", "first": "1"}]
    .   [13, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 13, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {}]
    .   [14, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 16, 1, "BREAKPOINT", "VERTICAL_FORWARD", {"i": "Variable is optimized away and not available.", "next": "Variable is optimized away and not available.", "second": "Variable is optimized away and not available.", "total": "0", "first": "Variable is optimized away and not available."}]
    .   [15, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 15, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {"i": "Variable is optimized away and not available.", "second": "2", "total": "0", "first": "1"}]
    .   [16, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 13, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {}]
    .   [17, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 16, 1, "BREAKPOINT", "VERTICAL_FORWARD", {"i": "Variable is optimized away and not available.", "next": "Variable is optimized away and not available.", "second": "Variable is optimized away and not available.", "total": "0", "first": "Variable is optimized away and not available."}]
    .   [18, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 15, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {"i": "Variable is optimized away and not available.", "second": "3", "total": "0", "first": "2"}]
    .   [19, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 13, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {}]
    .   [20, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 16, 1, "BREAKPOINT", "VERTICAL_FORWARD", {"i": "Variable is optimized away and not available.", "next": "Variable is optimized away and not available.", "second": "Variable is optimized away and not available.", "total": "0", "first": "Variable is optimized away and not available."}]
    .   [21, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 15, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {"i": "Variable is optimized away and not available.", "second": "5", "total": "0", "first": "3"}]
    .   [22, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 13, 1, "BREAKPOINT", "VERTICAL_BACKWARD", {}]
    .   [23, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 16, 1, "BREAKPOINT", "VERTICAL_FORWARD", {"i": "Variable is optimized away and not available.", "next": "Variable is optimized away and not available.", "second": "Variable is optimized away and not available.", "total": "0", "first": "Variable is optimized away and not available."}]
    .   [24, "Fibonacci", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 20, 1, "BREAKPOINT", "VERTICAL_FORWARD", {}]
    [25, "main", "c:\\dexter\\tests\\nostdlib\\fibonacci\\test.cpp", 26, 1, "BREAKPOINT", "FUNC", {"total": "7"}]
    ## END (25 steps) ##


    step kind differences [0/1]
        FUNC_EXTERNAL:
        0

    test.cpp:15-18 [first] [9/21]
        expected encountered values:
        0
        1
        2
        3

        missing values:
        5 [-6]

        result optimized away:
        step 5 (Variable is optimized away and not available.) [-3]
        step 7 (Variable is optimized away and not available.)
        step 8 (Variable is optimized away and not available.)
        step 11 (Variable is optimized away and not available.)
        step 14 (Variable is optimized away and not available.)
        step 17 (Variable is optimized away and not available.)
        step 20 (Variable is optimized away and not available.)
        step 23 (Variable is optimized away and not available.)

    test.cpp:15-18 [i] [15/21]
        result optimized away:
        step 5 (Variable is optimized away and not available.) [-3]
        step 7 (Variable is optimized away and not available.) [-3]
        step 8 (Variable is optimized away and not available.) [-3]
        step 9 (Variable is optimized away and not available.) [-3]
        step 11 (Variable is optimized away and not available.) [-3]
        step 12 (Variable is optimized away and not available.)
        step 14 (Variable is optimized away and not available.)
        step 15 (Variable is optimized away and not available.)
        step 17 (Variable is optimized away and not available.)
        step 18 (Variable is optimized away and not available.)
        step 20 (Variable is optimized away and not available.)
        step 21 (Variable is optimized away and not available.)
        step 23 (Variable is optimized away and not available.)

    test.cpp:15-18 [second] [21/21]
        expected encountered values:
        1
        2
        3
        5

        result optimized away:
        step 5 (Variable is optimized away and not available.) [-3]
        step 7 (Variable is optimized away and not available.) [-3]
        step 8 (Variable is optimized away and not available.) [-3]
        step 11 (Variable is optimized away and not available.) [-3]
        step 14 (Variable is optimized away and not available.) [-3]
        step 17 (Variable is optimized away and not available.) [-3]
        step 20 (Variable is optimized away and not available.) [-3]
        step 23 (Variable is optimized away and not available.)

    test.cpp:15-18 [total] [21/21]
        expected encountered values:
        0

        missing values:
        1 [-6]
        2 [-6]
        4 [-6]
        7 [-3]

    test.cpp:16-18 [next] [15/21]
        result optimized away:
        step 5 (Variable is optimized away and not available.) [-3]
        step 8 (Variable is optimized away and not available.) [-3]
        step 11 (Variable is optimized away and not available.) [-3]
        step 14 (Variable is optimized away and not available.) [-3]
        step 17 (Variable is optimized away and not available.) [-3]
        step 20 (Variable is optimized away and not available.)
        step 23 (Variable is optimized away and not available.)

    test.cpp:26 [total] [0/7]
        expected encountered values:
        7

The first line

    fibonacci =  (0.2832)

shows a score of 0.2832 suggesting that unexpected behavior has been seen.  This score is on scale of 0.0000 to 1.000, with 0.000 being the worst score possible and 1.000 being the best score possible.  The verbose output shows the reason for any scoring.  For example:

    test.cpp:15-18 [first] [9/21]
        expected encountered values:
        0
        1
        2
        3

        missing values:
        5 [-6]

        result optimized away:
        step 5 (Variable is optimized away and not available.) [-3]
        step 7 (Variable is optimized away and not available.)
        step 8 (Variable is optimized away and not available.)
        step 11 (Variable is optimized away and not available.)
        step 14 (Variable is optimized away and not available.)
        step 17 (Variable is optimized away and not available.)
        step 20 (Variable is optimized away and not available.)
        step 23 (Variable is optimized away and not available.)

shows that for `first` the expected values 0, 1, 2 and 3 were seen, 5 was not.  On some steps the variable was reported as being optimized away.

## Writing new test cases

Each test requires a `test.cfg` file.  Currently the contents of this file are not read, but its presence is used to determine the root directory of a test. In the future, configuration variables for the test such as supported language modes may be stored in this file. Use the various [commands](Commands.md) to encode debugging expectations.

## Additional tools

For clang-based compilers, the `clang-opt-bisect` tool can be used to get a breakdown of which LLVM passes may be contributing to debugging experience issues.  For example:

    $ dexter.py clang-opt-bisect tests/nostdlib/fibonacci --builder clang-cl --debugger vs2017 --cflags="/Ox /Zi" --ldflags="/Zi"

    pass 1/211 =  (1.0000)  (0.0000) [Simplify the CFG on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 2/211 =  (0.7611) (-0.2389) [SROA on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 3/211 =  (0.7611)  (0.0000) [Early CSE on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 4/211 =  (0.7611)  (0.0000) [Simplify the CFG on function (main)]
    pass 5/211 =  (0.7611)  (0.0000) [SROA on function (main)]
    pass 6/211 =  (0.7611)  (0.0000) [Early CSE on function (main)]
    pass 7/211 =  (0.7611)  (0.0000) [Infer set function attributes on module (c:\dexter\tests\fibonacci\test.cpp)]
    pass 8/211 =  (0.7611)  (0.0000) [Interprocedural Sparse Conditional Constant Propagation on module (c:\dexter\tests\fibonacci\test.cpp)]
    pass 9/211 =  (0.7611)  (0.0000) [Called Value Propagation on module (c:\dexter\tests\fibonacci\test.cpp)]
    pass 10/211 =  (0.7611)  (0.0000) [Global Variable Optimizer on module (c:\dexter\tests\fibonacci\test.cpp)]
    pass 11/211 =  (0.7611)  (0.0000) [Promote Memory to Register on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 12/211 =  (0.7611)  (0.0000) [Promote Memory to Register on function (main)]
    pass 13/211 =  (0.7611)  (0.0000) [Dead Argument Elimination on module (c:\dexter\tests\fibonacci\test.cpp)]
    pass 14/211 =  (0.7611)  (0.0000) [Combine redundant instructions on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 15/211 =  (0.7611)  (0.0000) [Simplify the CFG on function (?Fibonacci@@YAXHAEAH@Z)]a
    pass 16/211 =  (0.7345) (-0.0265) [Combine redundant instructions on function (main)]
    pass 17/211 =  (0.7345)  (0.0000) [Simplify the CFG on function (main)]
    pass 18/211 =  (0.7345)  (0.0000) [Remove unused exception handling info on SCC (?Fibonacci@@YAXHAEAH@Z)]
    pass 19/211 =  (0.7345)  (0.0000) [Function Integration/Inlining on SCC (?Fibonacci@@YAXHAEAH@Z)]
    pass 20/211 =  (0.7345)  (0.0000) [Deduce function attributes on SCC (?Fibonacci@@YAXHAEAH@Z)]
    pass 21/211 =  (0.7345)  (0.0000) [SROA on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 22/211 =  (0.7345)  (0.0000) [Early CSE w/ MemorySSA on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 23/211 =  (0.7345)  (0.0000) [Speculatively execute instructions if target has divergent branches on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 24/211 =  (0.7345)  (0.0000) [Jump Threading on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 25/211 =  (0.7345)  (0.0000) [Value Propagation on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 26/211 =  (0.7345)  (0.0000) [Simplify the CFG on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 27/211 =  (0.7345)  (0.0000) [Combine redundant instructions on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 28/211 =  (0.7345)  (0.0000) [Tail Call Elimination on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 29/211 =  (0.7345)  (0.0000) [Simplify the CFG on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 30/211 =  (0.7345)  (0.0000) [Reassociate expressions on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 31/211 =  (0.8673)  (0.1327) [Rotate Loops on loop]
    pass 32/211 =  (0.5575) (-0.3097) [Loop Invariant Code Motion on loop]
    pass 33/211 =  (0.5575)  (0.0000) [Unswitch loops on loop]
    pass 34/211 =  (0.5575)  (0.0000) [Simplify the CFG on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 35/211 =  (0.5575)  (0.0000) [Combine redundant instructions on function (?Fibonacci@@YAXHAEAH@Z)]
    pass 36/211 =  (0.5575)  (0.0000) [Induction Variable Simplification on loop]
    pass 37/211 =  (0.5575)  (0.0000) [Recognize loop idioms on loop]
    <output-snipped>

