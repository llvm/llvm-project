"""
Test that member lookup inside of generic container doesn't crash
"""
import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(
    __file__, globals(), decorators=[skipUnlessDarwin])
