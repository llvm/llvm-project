# Automating Stepping Logic

A slightly esoteric use of the Python API's is to construct custom stepping
types. LLDB's stepping is driven by a stack of "thread plans" and a fairly
simple state machine that runs the plans. You can create a Python class that
works as a thread plan, and responds to the requests the state machine makes to
run its operations.

The base class for the [ScriptedThreadPlan](https://lldb.llvm.org/python_api/lldb.plugins.scripted_thread_plan.ScriptedThreadPlan.html) is provided as part of the lldb python module, making it easy to derive a new class from it.

There is a longer discussion of scripted thread plans and the state machine,
and several interesting examples of their use in [scripted_step.py](https://github.com/llvm/llvm-project/blob/main/lldb/examples/python/scripted_step.py)
and for a **MUCH** fuller discussion of the whole state machine, see [ThreadPlan.h](https://github.com/llvm/llvm-project/blob/main/lldb/include/lldb/Target/ThreadPlan.h)

If you are reading those comments it is useful to know that scripted thread
plans are set to be either ***"ControllingPlans"*** or ***"OkayToDiscard"***.

To implement a scripted step, you define a python class that has the following
methods:

| Name | Arguments | Description |
|------|-----------|-------------|
| `__init__` | `thread_plan`: `lldb.SBThreadPlan` | This is the underlying `SBThreadPlan` that is pushed onto the plan stack. You will want to store this away in an ivar. Also, if you are going to use one of the canned thread plans, you can queue it at this point. |
| `explains_stop` | `event`: `lldb.SBEvent` | Return True if this stop is part of your thread plans logic, false otherwise. |
| `is_stale` | `None` | If your plan is no longer relevant (for instance, you were stepping in a particular stack frame, but some other operation pushed that frame off the stack) return True and your plan will get popped. |
| `should_step` | `None` | Return `True` if you want lldb to instruction step one instruction, or False to continue till the next breakpoint is hit. |
| `should_stop` | `event`: `lldb.SBEvent` | If your plan wants to stop and return control to the user at this point, return True. If your plan is done at this point, call SetPlanComplete on your thread plan instance. Also, do any work you need here to set up the next stage of stepping. |

To use this class to implement a step, use the command:

```python3
(lldb) thread step-scripted -C MyModule.MyStepPlanClass
```

Or use the `SBThread.StepUsingScriptedThreadPlan` API. The `SBThreadPlan` passed
into your `__init__` function can also push several common plans (step
in/out/over and run-to-address) in front of itself on the stack, which can be
used to compose more complex stepping operations. When you use subsidiary plans
your explains_stop and should_stop methods won't get called until the
subsidiary plan is done, or the process stops for an event the subsidiary plan
doesn't explain. For instance, step over plans don't explain a breakpoint hit
while performing the step-over.