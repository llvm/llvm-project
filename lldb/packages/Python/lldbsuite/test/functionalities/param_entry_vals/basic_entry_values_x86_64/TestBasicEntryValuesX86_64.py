from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(__file__, globals(),
        [decorators.skipUnlessDarwin,
         decorators.skipUnlessArch('x86_64'),
         decorators.skipUnlessHasCallSiteInfo,
         decorators.skipIf(dwarf_version=['<', '4'])])
