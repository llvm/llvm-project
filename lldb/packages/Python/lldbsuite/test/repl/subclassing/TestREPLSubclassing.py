import lldbsuite.test.lldbinrepl as lldbinrepl
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.decorators as decorators

lldbinrepl.MakeREPLTest(__file__, globals(), decorators=[
    decorators.skipIf]) # rdar://36843869
