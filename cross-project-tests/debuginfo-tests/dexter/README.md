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

The following commands build fibonacci.cpp from the tests/nostdlib directory and run it in LLDB, reporting the debug experience heuristic. The first pair of commands build with no optimizations (-O0) and score 1.0000.  The second pair of commands build with optimizations (-O2) and score 0.2832 which suggests a worse debugging experience.

    clang -O0 -g tests/nostdlib/fibonacci.cpp -o tests/nostdlib/fibonacci/test
    dexter.py test --binary tests/nostdlib/fibonacci/test --debugger lldb -- tests/nostdlib/fibonacci/test.cpp
    test.cpp = (1.0000)

    clang -O2 -g tests/nostdlib/fibonacci/test.cpp -o tests/nostdlib/fibonacci/test
    dexter.py test --binary tests/nostdlib/fibonacci/test --debugger lldb -- tests/nostdlib/fibonacci/test.cpp
    test.cpp = (0.2832)

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

Running the command below launches the tests/nostdlib/fibonacci test case in DExTer, using LLDB as the debugger and producing a detailed report:

    $ dexter.py test --vs-solution clang-cl_vs2015 --debugger vs2017 --cflags="/Ox /Zi" --ldflags="/Zi" -v -- tests/nostdlib/fibonacci

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

Each test can be either embedded within the source file using comments or included as a separate file with the .dex extension. Dexter does not include support for building test cases, although if a Visual Studio Solution (.sln) is used as the test file, VS will build the program as part of launching a debugger session if it has not already been built.
