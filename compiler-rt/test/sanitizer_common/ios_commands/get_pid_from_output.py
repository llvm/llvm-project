"""
Parses the id of the process that ran with ASAN from the output logs.
"""
import sys, argparse, re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='The sanitizer output to get the pid from')
    parser.add_argument('--outfile', nargs='?', type=argparse.FileType('r'), default=sys.stdout, help='Where to write the result')
    args = parser.parse_args()

    pid = process_file(args.infile)
    args.outfile.write(pid)
    args.infile.close()
    args.outfile.close()



def process_file(infile):
    # check first line is just ==== divider
    first_line_pattern = re.compile(r'=*')
    assert first_line_pattern.match(infile.readline())

    # parse out pid from 2nd line 
    # `==PID==ERROR: SanitizerName: error-type on address...`
    pid_pattern = re.compile(r'==([0-9]*)==ERROR:')
    pid = pid_pattern.search(infile.readline()).group(1)

    # ignore the rest

    assert pid and pid.isdigit()

    return pid

if __name__ == '__main__':
    main()
