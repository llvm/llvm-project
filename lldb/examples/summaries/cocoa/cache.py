"""
Objective-C runtime wrapper for use by LLDB Python formatters

part of The LLVM Compiler Infrastructure
This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.
"""
import lldb.formatters.metrics


class Cache:

    def __init__(self):
        self.data = {}
        self.statistics = lldb.formatters.metrics.Metrics()
        self.statistics.add_metric('hit')
        self.statistics.add_metric('miss')

    def look_for_key(self, key):
        if key in self.data:
            return True
        return False

    def add_item(self, key, value, ok_to_replace=True):
        if not(ok_to_replace) and self.look_for_key(key):
            return False
        self.data[key] = value
        return True

    def get_value(self, key, default=None):
        if self.look_for_key(key):
            self.statistics.metric_hit('hit', key)
            return self.data[key]
        else:
            self.statistics.metric_hit('miss', key)
            return default
