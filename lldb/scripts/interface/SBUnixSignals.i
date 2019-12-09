//===-- SWIG Interface for SBUnixSignals ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Allows you to manipulate LLDB's signal disposition"
) SBUnixSignals;
class SBUnixSignals
{
public:
    SBUnixSignals ();

    SBUnixSignals (const lldb::SBUnixSignals &rhs);

    ~SBUnixSignals();

    void
    Clear ();

    bool
    IsValid () const;

    explicit operator bool() const;

    const char *
    GetSignalAsCString (int32_t signo) const;

    int32_t
    GetSignalNumberFromName (const char *name) const;

    bool
    GetShouldSuppress (int32_t signo) const;

    bool
    SetShouldSuppress (int32_t signo,
                       bool value);

    bool
    GetShouldStop (int32_t signo) const;

    bool
    SetShouldStop (int32_t signo,
                   bool value);

    bool
    GetShouldNotify (int32_t signo) const;

    bool
    SetShouldNotify (int32_t signo, bool value);

    int32_t
    GetNumSignals () const;

    int32_t
    GetSignalAtIndex (int32_t index) const;

#ifdef SWIGPYTHON
    %pythoncode %{
        def get_unix_signals_list(self):
            signals = []
            for idx in range(0, self.GetNumSignals()):
                signals.append(self.GetSignalAtIndex(sig))
            return signals

        threads = property(get_unix_signals_list, None, doc='''A read only property that returns a list() of valid signal numbers for this platform.''')
    %}
#endif
};

}  // namespace lldb
