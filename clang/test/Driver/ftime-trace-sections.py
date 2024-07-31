#!/usr/bin/env python

import json, sys, time


def is_inside(range1, range2):
    a = range1["ts"]
    b = a + range1["dur"]
    c = range2["ts"]
    d = c + range2["dur"]
    return (a >= c and a <= d) and (b >= c and b <= d)


def is_before(range1, range2):
    b = range1["ts"] + range1["dur"]
    c = range2["ts"]
    return b <= c


log_contents = json.loads(sys.stdin.read())
events = log_contents["traceEvents"]

instants = [event for event in events if event["name"] == "InstantiateFunction"]
codegens = [event for event in events if event["name"] == "CodeGen Function"]
opts = [event for event in events if event["name"] == "OptFunction"]
frontends = [event for event in events if event["name"] == "Frontend"]
backends = [event for event in events if event["name"] == "Backend"]

beginning_of_time = log_contents["beginningOfTime"] / 1000000
seconds_since_epoch = time.time()

# Make sure that the 'beginningOfTime' is not later than now.
if beginning_of_time > seconds_since_epoch:
    sys.exit(
        "'beginningOfTime' should represent the absolute time when the "
        "process has started"
    )

if not all(
    [
        any([is_inside(codegen, frontend) for frontend in frontends])
        for codegen in codegens
    ]
):
    sys.exit("Not all CodeGen sections are inside any Frontend section!")

if not all(
    [
        all([is_before(frontend, backend) for frontend in frontends])
        for backend in backends
    ]
):
    sys.exit("Not all Frontend section are before all Backend sections!")

# Check that entries for foo exist and are in a demangled form.
if not any(e for e in instants if "foo<int>" in e["args"]["detail"]):
    sys.exit("Missing Instantiate entry for foo!")
if not any(e for e in codegens if "foo<int>" in e["args"]["detail"]):
    sys.exit("Missing CodeGen entry for foo!")
if not any(e for e in opts if "foo<int>" in e["args"]["detail"]):
    sys.exit("Missing Optimize entry for foo!")
