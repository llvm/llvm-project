//====-- UserSettingsController.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string.h>
#include <algorithm>

#include "lldb/Core/UserSettingsController.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/OptionValueString.h"

using namespace lldb;
using namespace lldb_private;


lldb::OptionValueSP
Properties::GetPropertyValue (const ExecutionContext *exe_ctx,
                              const char *path,
                              bool will_modify,
                              Error &error) const
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
        return properties_sp->GetSubValue(exe_ctx, path, will_modify, error);
    return lldb::OptionValueSP();
}

Error
Properties::SetPropertyValue (const ExecutionContext *exe_ctx,
                              VarSetOperationType op,
                              const char *path,
                              const char *value)
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
        return properties_sp->SetSubValue(exe_ctx, op, path, value);
    Error error;
    error.SetErrorString ("no properties");
    return error;
}

void
Properties::DumpAllPropertyValues (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask)
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
        return properties_sp->DumpValue (exe_ctx, strm, dump_mask);
}

void
Properties::DumpAllDescriptions (CommandInterpreter &interpreter,
                                 Stream &strm) const
{
    strm.PutCString("Top level variables:\n\n");

    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
        return properties_sp->DumpAllDescriptions (interpreter, strm);
}



Error
Properties::DumpPropertyValue (const ExecutionContext *exe_ctx, Stream &strm, const char *property_path, uint32_t dump_mask)
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
    {
        return properties_sp->DumpPropertyValue (exe_ctx,
                                                 strm,
                                                 property_path,
                                                 dump_mask);
    }
    Error error;
    error.SetErrorString("empty property list");
    return error;
}

size_t
Properties::Apropos (const char *keyword, std::vector<const Property *> &matching_properties) const
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
    {
        properties_sp->Apropos (keyword, matching_properties);
    }
    return matching_properties.size();
}


lldb::OptionValuePropertiesSP
Properties::GetSubProperty (const ExecutionContext *exe_ctx,
                            const ConstString &name)
{
    OptionValuePropertiesSP properties_sp (GetValueProperties ());
    if (properties_sp)
        return properties_sp->GetSubProperty (exe_ctx, name);
    return lldb::OptionValuePropertiesSP();
}

