//===-- SWIG Interface for SBCommandInterpreter -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"SBCommandInterpreterRunOptions controls how the RunCommandInterpreter runs the code it is fed.

A default SBCommandInterpreterRunOptions object has:

* StopOnContinue: false
* StopOnError:    false
* StopOnCrash:    false
* EchoCommands:   true
* PrintResults:   true
* PrintErrors:    true
* AddToHistory:   true

") SBCommandInterpreterRunOptions;
class SBCommandInterpreterRunOptions
{
friend class SBDebugger;
public:
    SBCommandInterpreterRunOptions();
    ~SBCommandInterpreterRunOptions();

    bool
    GetStopOnContinue () const;

    void
    SetStopOnContinue (bool);

    bool
    GetStopOnError () const;

    void
    SetStopOnError (bool);

    bool
    GetStopOnCrash () const;

    void
    SetStopOnCrash (bool);

    bool
    GetEchoCommands () const;

    void
    SetEchoCommands (bool);

    bool
    GetPrintResults () const;

    void
    SetPrintResults (bool);

    bool
    GetPrintErrors () const;

    void
    SetPrintErrors (bool);

    bool
    GetAddToHistory () const;

    void
    SetAddToHistory (bool);
private:
    lldb_private::CommandInterpreterRunOptions *
    get () const;

    lldb_private::CommandInterpreterRunOptions &
    ref () const;

    // This is set in the constructor and will always be valid.
    mutable std::unique_ptr<lldb_private::CommandInterpreterRunOptions> m_opaque_up;
};

} // namespace lldb
