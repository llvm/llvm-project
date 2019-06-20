import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(__file__, globals(), decorators=[swiftTest, skipIf(macos_version=["<", "10.15"])])
