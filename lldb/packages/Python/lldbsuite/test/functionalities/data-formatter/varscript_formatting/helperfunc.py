import lldb


def f(value, d):
    return "pointer type" if value.GetType().GetTemplateArgumentType(
        0).IsPointerType() else "non-pointer type"
