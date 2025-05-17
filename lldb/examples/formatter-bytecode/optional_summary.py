def OptionalSummaryProvider(valobj, _):
    failure = 2
    storage = valobj.GetChildMemberWithName("Storage")
    hasVal = storage.GetChildMemberWithName("hasVal").GetValueAsUnsigned(failure)
    if hasVal == failure:
        return "<could not read Optional>"

    if hasVal == 0:
        return "None"

    underlying_type = storage.GetType().GetTemplateArgumentType(0)
    value = storage.GetChildMemberWithName("value")
    value = value.Cast(underlying_type)
    return value.GetSummary()
