# encoding: utf-8

import lldb

class MyFrameRecognizer(object):
    def get_recognized_arguments(self, frame):
        if frame.name == "foo":
            arg1 = frame.EvaluateExpression("$arg1").signed
            arg2 = frame.EvaluateExpression("$arg2").signed
            val1 = lldb.target.CreateValueFromExpression("a", "%d" % arg1)
            val2 = lldb.target.CreateValueFromExpression("b", "%d" % arg2)
            return [val1, val2]
        elif frame.name == "bar":
            arg1 = frame.EvaluateExpression("$arg1").signed
            val1 = lldb.target.CreateValueFromExpression("a", "(int *)%d" % arg1)
            return [val1]
        return []

class MyOtherFrameRecognizer(object):
    def get_recognized_arguments(self, frame):
        return []
