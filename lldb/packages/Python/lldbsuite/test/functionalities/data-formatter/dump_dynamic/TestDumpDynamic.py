import lldbsuite.test.lldbinline as lldbinline

lldbinline.MakeInlineTest(__file__, globals(), [lldbinline.expectedFailureWindows("llvm.org/pr24663")])
