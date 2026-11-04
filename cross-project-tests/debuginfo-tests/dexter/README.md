# DExTer (Debugging Experience Tester)

## Introduction

DExTer is a suite of tools used to evaluate the "User Debugging Experience". DExTer drives an external debugger, running on small test programs, and collects information on the behavior at each debugger step to provide quantitative values that indicate the quality of the debugging experience.

## Supported Debuggers

DExTer currently supports the Visual Studio 2015 and Visual Studio 2017 debuggers via the [DTE interface](https://docs.microsoft.com/en-us/dotnet/api/envdte.dte), and LLDB via its [Python interface](https://lldb.llvm.org/python-reference.html) and its [DAP interface](https://lldb.llvm.org/use/lldbdap.html). GDB is not currently supported.

The following command evaluates your environment, listing the available and compatible debuggers:

    dexter.py list-debuggers

## Dependencies

See: requirements.txt

### Python 3.8

DExTer requires python version 3.8 or greater.

### pywin32 python package

This is required to access the DTE interface for the Visual Studio debuggers.

    <python-executable> -m pip install pywin32

## Usage

Dexter has two distinct usage modes: "heuristic" mode and "script" mode. The heuristic mode is the old/original mode for Dexter, using a set of declarative commands to control the debug session, fetch information, and produce a single debug experience heuristic score as the final output. The script mode is the new/current mode used by Dexter, which uses a structured YAML script to control the debug session and fetch information, and produces as its output a list of metrics which collectively describe the debug experience. 

For more information, see:
### [Scripts.md](./Script.md)
### [Heuristic.md](./Heuristic.md)
