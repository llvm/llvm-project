from __future__ import print_function


def function(frame, bp_loc, dict):
    there = open("output2.txt", "w")
    print("lldb", file=there)
    there.close()
