#!/usr/bin/env python

import json, sys

def is_inside(range1, range2):
    a = range1["ts"]; b = a + range1["dur"]
    c = range2["ts"]; d = c + range2["dur"]
    return (a >= c and a <= d) and (b >= c and b <= d)

def is_before(range1, range2):
    b = range1["ts"] + range1["dur"]; c = range2["ts"]
    return b <= c

events = json.loads(sys.stdin.read())["traceEvents"]
codegens = filter(lambda x: x["name"] == "CodeGen Function", events)
frontends = filter(lambda x: x["name"] == "Frontend", events)
backends = filter(lambda x: x["name"] == "Backend", events)

if not all([any([is_inside(codegen, frontend) for frontend in frontends])
                        for codegen in codegens]):
    sys.exit("Not all CodeGen sections are inside any Frontend section!")

if not all([all([is_before(frontend, backend) for frontend in frontends])
                        for backend in backends]):
    sys.exit("Not all Frontend section are before all Backend sections!")
