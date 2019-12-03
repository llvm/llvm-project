from lldbsuite.test import lldbinline
from lldbsuite.test import decorators
from lldbsuite.test import lldbplatformutil

supported_platforms = ["linux"]
supported_platforms.extend(lldbplatformutil.getDarwinOSTriples())

lldbinline.MakeInlineTest(__file__, globals(),
        [decorators.skipUnlessPlatform(supported_platforms),
         decorators.skipIf(compiler="clang", compiler_version=['<', '10.0']),
         decorators.skipUnlessArch('x86_64'),
         decorators.skipUnlessHasCallSiteInfo,
         decorators.skipIf(dwarf_version=['<', '4'])])
