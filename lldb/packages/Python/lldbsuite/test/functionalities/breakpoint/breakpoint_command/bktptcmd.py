from __future__ import print_function
import side_effect

def function(frame, bp_loc, dict):
    side_effect.bktptcmd = "function was here"
