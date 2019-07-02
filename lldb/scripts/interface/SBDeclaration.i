//===-- SWIG Interface for SBDeclaration --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

%include <attribute.i>

namespace lldb {

    %feature("docstring",
    "Specifies an association with a line and column for a variable."
    ) SBDeclaration;
    class SBDeclaration
    {
        public:

        SBDeclaration ();

        SBDeclaration (const lldb::SBDeclaration &rhs);

        ~SBDeclaration ();

        bool
        IsValid () const;

        explicit operator bool() const;

        lldb::SBFileSpec
        GetFileSpec () const;

        uint32_t
        GetLine () const;

        uint32_t
        GetColumn () const;

        bool
        GetDescription (lldb::SBStream &description);

        void
        SetFileSpec (lldb::SBFileSpec filespec);

        void
        SetLine (uint32_t line);

        void
        SetColumn (uint32_t column);

        bool
        operator == (const lldb::SBDeclaration &rhs) const;

        bool
        operator != (const lldb::SBDeclaration &rhs) const;
    };

%attributeref(lldb::SBDeclaration, lldb::SBFileSpec, file, GetFileSpec);
%attribute(lldb::SBDeclaration, uint32_t, line, GetLine);
%attribute(lldb::SBDeclaration, uint32_t, column, GetColumn);

} // namespace lldb
