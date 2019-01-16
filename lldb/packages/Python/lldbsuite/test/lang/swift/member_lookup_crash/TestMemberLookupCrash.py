"""
Test that member lookup inside of generic container doesn't crash
"""
import lldbsuite.test.lldbinline as lldbinline
import lldbsuite.test.decorators as decorators

lldbinline.MakeInlineTest(
    __file__, globals(), decorators=[decorators.skipUnlessDarwin])
