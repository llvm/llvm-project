import lldb

class StepWithChild:
    def __init__(self, thread_plan):
        self.thread_plan = thread_plan
        self.child_thread_plan = self.queue_child_thread_plan()

    def explains_stop(self, event):
        return False

    def should_stop(self, event):
        if not self.child_thread_plan.IsPlanComplete():
            return False

        self.thread_plan.SetPlanComplete(True)

        return True

    def should_step(self):
        return False

    def queue_child_thread_plan(self):
        return None

class StepOut(StepWithChild):
    def __init__(self, thread_plan, dict):
        StepWithChild.__init__(self, thread_plan)

    def queue_child_thread_plan(self):
        return self.thread_plan.QueueThreadPlanForStepOut(0)

class StepScripted(StepWithChild):
    def __init__(self, thread_plan, dict):
        StepWithChild.__init__(self, thread_plan)

    def queue_child_thread_plan(self):
        return self.thread_plan.QueueThreadPlanForStepScripted("Steps.StepOut")
