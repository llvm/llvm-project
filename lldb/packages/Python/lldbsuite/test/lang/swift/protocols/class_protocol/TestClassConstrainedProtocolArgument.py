"""
Test that variables passed in as a class constrained protocol type
are correctly printed.
"""
import lldbsuite.test.lldbinline as lldbinline
import lldbsuite.test.decorators as decorators

lldbinline.MakeInlineTest(
    __file__, globals(), decorators=[decorators.skipUnlessDarwin])
