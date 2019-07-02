//===-- SWIG Interface for SBTypeSynthetic-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

    %feature("docstring",
    "Represents a summary that can be associated to one or more types.") SBTypeSynthetic;

    class SBTypeSynthetic
    {
    public:

        SBTypeSynthetic();

        static lldb::SBTypeSynthetic
        CreateWithClassName (const char* data, uint32_t options = 0);

        static lldb::SBTypeSynthetic
        CreateWithScriptCode (const char* data, uint32_t options = 0);

        SBTypeSynthetic (const lldb::SBTypeSynthetic &rhs);

        ~SBTypeSynthetic ();

        bool
        IsValid() const;

        explicit operator bool() const;

        bool
        IsEqualTo (lldb::SBTypeSynthetic &rhs);

        bool
        IsClassCode();

        const char*
        GetData ();

        void
        SetClassName (const char* data);

        void
        SetClassCode (const char* data);

        uint32_t
        GetOptions ();

        void
        SetOptions (uint32_t);

        bool
        GetDescription (lldb::SBStream &description,
                        lldb::DescriptionLevel description_level);

        bool
        operator == (lldb::SBTypeSynthetic &rhs);

        bool
        operator != (lldb::SBTypeSynthetic &rhs);

        %pythoncode %{
            options = property(GetOptions, SetOptions)
            contains_code = property(IsClassCode, None)
            synthetic_data = property(GetData, None)
        %}

    };

} // namespace lldb
