//===-- SWIG Interface for SBCommandReturnObject ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a container which holds the result from command execution.
It works with SBCommandInterpreter.HandleCommand() to encapsulate the result
of command execution.

See SBCommandInterpreter for example usage of SBCommandReturnObject."
) SBCommandReturnObject;
class SBCommandReturnObject
{
public:

    SBCommandReturnObject ();

    SBCommandReturnObject (const lldb::SBCommandReturnObject &rhs);

    ~SBCommandReturnObject ();

    bool
    IsValid() const;

    explicit operator bool() const;

    const char *
    GetOutput ();

    const char *
    GetError ();

    size_t
    GetOutputSize ();

    size_t
    GetErrorSize ();

    const char *
    GetOutput (bool only_if_no_immediate);

    const char *
    GetError (bool if_no_immediate);

    size_t
    PutOutput (lldb::SBFile file);

    size_t
    PutError (lldb::SBFile file);

    size_t
    PutOutput (lldb::FileSP BORROWED);

    size_t
    PutError (lldb::FileSP BORROWED);

    void
    Clear();

    void
    SetStatus (lldb::ReturnStatus status);

    void
    SetError (lldb::SBError &error,
              const char *fallback_error_cstr = NULL);

    void
    SetError (const char *error_cstr);

    lldb::ReturnStatus
    GetStatus();

    bool
    Succeeded ();

    bool
    HasResult ();

    void
    AppendMessage (const char *message);

    void
    AppendWarning (const char *message);

    bool
    GetDescription (lldb::SBStream &description);

    void SetImmediateOutputFile(lldb::SBFile file);
    void SetImmediateErrorFile(lldb::SBFile file);
    void SetImmediateOutputFile(lldb::FileSP BORROWED);
    void SetImmediateErrorFile(lldb::FileSP BORROWED);

    %extend {
        // transfer_ownership does nothing, and is here for compatibility with
        // old scripts.  Ownership is tracked by reference count in the ordinary way.

        void SetImmediateOutputFile(lldb::FileSP BORROWED, bool transfer_ownership) {
            self->SetImmediateOutputFile(BORROWED);
        }
        void SetImmediateErrorFile(lldb::FileSP BORROWED, bool transfer_ownership) {
            self->SetImmediateErrorFile(BORROWED);
        }
    }

	void
	PutCString(const char* string, int len);

    // wrapping the variadic Printf() with a plain Print()
    // because it is hard to support varargs in SWIG bridgings
    %extend {
        void Print (const char* str)
        {
            self->Printf("%s", str);
        }
    }

};

} // namespace lldb
