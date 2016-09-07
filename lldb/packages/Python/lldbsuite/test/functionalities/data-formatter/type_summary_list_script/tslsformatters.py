import lldb


def Struct_SummaryFormatter(valobj, internal_dict):
    return 'A data formatter at work'

category = lldb.debugger.CreateCategory("TSLSFormatters")
category.SetEnabled(True)
summary = lldb.SBTypeSummary.CreateWithFunctionName(
    "tslsformatters.Struct_SummaryFormatter", lldb.eTypeOptionCascade)
spec = lldb.SBTypeNameSpecifier("Struct", False)
category.AddTypeSummary(spec, summary)
