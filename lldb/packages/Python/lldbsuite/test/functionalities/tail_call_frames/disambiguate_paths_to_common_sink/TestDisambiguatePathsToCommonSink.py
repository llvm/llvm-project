from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(__file__, globals(),
        [decorators.skipUnlessHasCallSiteInfo,
         decorators.skipIf(dwarf_version=['<', '4'])])
