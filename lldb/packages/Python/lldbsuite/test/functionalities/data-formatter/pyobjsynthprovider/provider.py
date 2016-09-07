import lldb
import lldb.formatters
import lldb.formatters.synth


class SyntheticChildrenProvider(
        lldb.formatters.synth.PythonObjectSyntheticChildProvider):

    def __init__(self, value, internal_dict):
        lldb.formatters.synth.PythonObjectSyntheticChildProvider.__init__(
            self, value, internal_dict)

    def make_children(self):
        return [("ID", 123456),
                ("Name", "Enrico"),
                ("Rate", 1.25)]
