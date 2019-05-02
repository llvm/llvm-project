//===-- SWIG Interface for SBTypeFilter----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

    %feature("docstring",
    "Represents a filter that can be associated to one or more types.") SBTypeFilter;

    class SBTypeFilter
    {
        public:

        SBTypeFilter();

        SBTypeFilter (uint32_t options);

        SBTypeFilter (const lldb::SBTypeFilter &rhs);

        ~SBTypeFilter ();

        bool
        IsValid() const;

        explicit operator bool() const;

        bool
        IsEqualTo (lldb::SBTypeFilter &rhs);

        uint32_t
        GetNumberOfExpressionPaths ();

        const char*
        GetExpressionPathAtIndex (uint32_t i);

        bool
        ReplaceExpressionPathAtIndex (uint32_t i, const char* item);

        void
        AppendExpressionPath (const char* item);

        void
        Clear();

        uint32_t
        GetOptions();

        void
        SetOptions (uint32_t);

        bool
        GetDescription (lldb::SBStream &description, lldb::DescriptionLevel description_level);

        bool
        operator == (lldb::SBTypeFilter &rhs);

        bool
        operator != (lldb::SBTypeFilter &rhs);

        %pythoncode %{
            __swig_getmethods__["options"] = GetOptions
            __swig_setmethods__["options"] = SetOptions
            if _newclass: options = property(GetOptions, SetOptions)

            __swig_getmethods__["count"] = GetNumberOfExpressionPaths
            if _newclass: count = property(GetNumberOfExpressionPaths, None)
        %}

    };

} // namespace lldb
