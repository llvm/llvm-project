===================
Analysis Statistics
===================

CSA enjoys two facilities to collect statistics: per translation unit and per entry point.
We use llvm/ADT/Statistic.h for numbers describing the entire translation unit (TU).
We use clang/StaticAnalyzer/Core/PathSensitive/EntryPointStats.h to collect data for each symbolic-execution entry point.

In many cases, it makes sense to collect statistics on both translation-unit level and entry-point level. You can use the two macros defined in EntryPointStats.h for that:

- ``STAT_COUNTER`` for additive statistics, for example, "the number of steps executed", "the number of functions inlined".
- ``STAT_MAX`` for maximizing statistics, for example, "the maximum environment size", or "the longest execution path".

If you want to define a statistic that makes sense only for the entire translation unit, for example, "the number of entry points", Statistic.h defines two macros: ``STATISTIC`` and ``ALWAYS_ENABLED_STATISTIC``.
You should prefer ``ALWAYS_ENABLED_STATISTIC`` unless you have a good reason not to.
``STATISTIC`` is controlled by ``LLVM_ENABLE_STATS`` / ``LLVM_FORCE_ENABLE_STATS``.
However, note that with ``LLVM_ENABLE_STATS`` disabled, only storage of the values is disabled, the computations producing those values still carry on unless you took an explicit precaution to make them conditional too.

If you want to define a statistic only for entry point, EntryPointStats.h has four classes at your disposal:


- ``BoolEPStat`` - a boolean value assigned at most once per entry point. For example: "has the inline limit been reached".
- ``UnsignedEPStat`` - an unsigned value assigned at most once per entry point. For example: "the number of source characters in an entry-point body".
- ``CounterEPStat`` - an additive statistic. It starts with 0 and you can add to it as many times as needed. For example: "the number of bugs discovered".
- ``UnsignedMaxEPStat`` - a maximizing statistic. It starts with 0 and when you join it with a value, it picks the maximum of the previous value and the new one. For example, "the longest execution path of a bug".

To produce a CSV file with all the statistics collected per entry point, use the `dump-entry-point-stats-to-csv=<file>.csv` parameter.
