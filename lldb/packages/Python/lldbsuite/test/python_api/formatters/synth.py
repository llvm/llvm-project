import lldb


class jasSynthProvider:

    def __init__(self, valobj, dict):
        self.valobj = valobj

    def num_children(self):
        return 2

    def get_child_at_index(self, index):
        child = None
        if index == 0:
            child = self.valobj.GetChildMemberWithName('A')
        if index == 1:
            child = self.valobj.CreateValueFromExpression('X', '(int)1')
        return child

    def get_child_index(self, name):
        if name == 'A':
            return 0
        if name == 'X':
            return 1
        return None


def ccc_summary(sbvalue, internal_dict):
    sbvalue = sbvalue.GetNonSyntheticValue()
    # This tests that the SBValue.GetNonSyntheticValue() actually returns a
    # non-synthetic value. If it does not, then sbvalue.GetChildMemberWithName("a")
    # in the following statement will call the 'get_child_index' method of the
    # synthetic child provider CCCSynthProvider below (which raises an
    # exception).
    return "CCC object with leading value " + \
        str(sbvalue.GetChildMemberWithName("a"))


class CCCSynthProvider(object):

    def __init__(self, sbvalue, internal_dict):
        self._sbvalue = sbvalue

    def num_children(self):
        return 3

    def get_child_index(self, name):
        raise RuntimeError("I don't want to be called!")

    def get_child_at_index(self, index):
        if index == 0:
            return self._sbvalue.GetChildMemberWithName("a")
        if index == 1:
            return self._sbvalue.GetChildMemberWithName("b")
        if index == 2:
            return self._sbvalue.GetChildMemberWithName("c")


def empty1_summary(sbvalue, internal_dict):
    return "I am an empty Empty1"


class Empty1SynthProvider(object):

    def __init__(self, sbvalue, internal_dict):
        self._sbvalue = sbvalue

    def num_children(self):
        return 0

    def get_child_at_index(self, index):
        return None


def empty2_summary(sbvalue, internal_dict):
    return "I am an empty Empty2"


class Empty2SynthProvider(object):

    def __init__(self, sbvalue, internal_dict):
        self._sbvalue = sbvalue

    def num_children(self):
        return 0

    def get_child_at_index(self, index):
        return None


def __lldb_init_module(debugger, dict):
    debugger.CreateCategory("JASSynth").AddTypeSynthetic(
        lldb.SBTypeNameSpecifier("JustAStruct"),
        lldb.SBTypeSynthetic.CreateWithClassName("synth.jasSynthProvider"))
    cat = lldb.debugger.CreateCategory("CCCSynth")
    cat.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier("CCC"),
        lldb.SBTypeSynthetic.CreateWithClassName("synth.CCCSynthProvider",
                                                 lldb.eTypeOptionCascade))
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier("CCC"),
        lldb.SBTypeSummary.CreateWithFunctionName("synth.ccc_summary",
                                                  lldb.eTypeOptionCascade))
    cat.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier("Empty1"),
        lldb.SBTypeSynthetic.CreateWithClassName("synth.Empty1SynthProvider"))
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier("Empty1"),
        lldb.SBTypeSummary.CreateWithFunctionName("synth.empty1_summary"))
    cat.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier("Empty2"),
        lldb.SBTypeSynthetic.CreateWithClassName("synth.Empty2SynthProvider"))
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier("Empty2"),
        lldb.SBTypeSummary.CreateWithFunctionName(
            "synth.empty2_summary",
            lldb.eTypeOptionHideEmptyAggregates))
