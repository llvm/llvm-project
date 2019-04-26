//===-- SWIG Interface for SBTypeFormat----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

    %feature("docstring",
             "Represents a format that can be associated to one or more types.") SBTypeFormat;

    class SBTypeFormat
    {
    public:

        SBTypeFormat();

        SBTypeFormat (lldb::Format format, uint32_t options = 0);

        SBTypeFormat (const char* type, uint32_t options = 0);

        SBTypeFormat (const lldb::SBTypeFormat &rhs);

        ~SBTypeFormat ();

        bool
        IsValid() const;

        explicit operator bool() const;

        bool
        IsEqualTo (lldb::SBTypeFormat &rhs);

        lldb::Format
        GetFormat ();

        const char*
        GetTypeName ();

        uint32_t
        GetOptions();

        void
        SetFormat (lldb::Format);

        void
        SetTypeName (const char*);

        void
        SetOptions (uint32_t);

        bool
        GetDescription (lldb::SBStream &description,
                        lldb::DescriptionLevel description_level);

        bool
        operator == (lldb::SBTypeFormat &rhs);

        bool
        operator != (lldb::SBTypeFormat &rhs);

        %pythoncode %{
            __swig_getmethods__["format"] = GetFormat
            __swig_setmethods__["format"] = SetFormat
            if _newclass: format = property(GetFormat, SetFormat)

            __swig_getmethods__["options"] = GetOptions
            __swig_setmethods__["options"] = SetOptions
            if _newclass: options = property(GetOptions, SetOptions)
        %}

    };


} // namespace lldb

