#!/usr/bin/env python
"""
compare_bench.py - Compare two benchmarks or their results and report the
                   difference.
"""
import sys
import gbench
from gbench import util, report

def main():
    # Parse the command line flags
    def usage():
        print('compare_bench.py <test1> <test2> [benchmark options]...')
        exit(1)
    if '--help' in sys.argv or len(sys.argv) < 3:
        usage()
    tests = sys.argv[1:3]
    bench_opts = sys.argv[3:]
    bench_opts = list(bench_opts)
    # Run the benchmarks and report the results
    json1 = gbench.util.run_or_load_benchmark(tests[0], bench_opts)
    json2 = gbench.util.run_or_load_benchmark(tests[1], bench_opts)
    output_lines = gbench.report.generate_difference_report(json1, json2)
    print 'Comparing %s to %s' % (tests[0], tests[1])
    for ln in output_lines:
        print(ln)


if __name__ == '__main__':
    main()
