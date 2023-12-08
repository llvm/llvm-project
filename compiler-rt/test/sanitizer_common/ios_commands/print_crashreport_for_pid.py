"""
Finds and prints the crash report associated with a specific (binary filename, process id).
Waits (max_wait_time/attempts_remaining) between retries. 
By default, max_wait_time=5 and retry_count=10, which results in a total wait time of ~15s
Errors if the report cannot be found after `retry_count` retries.
"""
import sys, os, argparse, re, glob, shutil, time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pid",
        type=str,
        required=True,
        help="The process id of the process that crashed",
    )
    parser.add_argument(
        "--binary-filename",
        type=str,
        required=True,
        help="The name of the file that crashed",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        nargs="?",
        default=10,
        help="The number of retries to make",
    )
    parser.add_argument(
        "--max-wait-time",
        type=float,
        nargs="?",
        default=5.0,
        help="The max amount of seconds to wait between tries",
    )

    parser.add_argument(
        "--dir",
        nargs="?",
        type=str,
        default="~/Library/Logs/DiagnosticReports",
        help="The directory to look for the crash report",
    )
    parser.add_argument(
        "--outfile",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdout,
        help="Where to write the result",
    )
    args = parser.parse_args()

    assert args.pid, "pid can't be empty"
    assert args.binary_filename, "binary-filename can't be empty"

    os.chdir(os.path.expanduser(args.dir))
    output_report_with_retries(
        args.outfile,
        args.pid.strip(),
        args.binary_filename,
        args.retry_count,
        args.max_wait_time,
    )


def output_report_with_retries(
    outfile, pid, filename, attempts_remaining, max_wait_time
):
    report_name = find_report_in_cur_dir(pid, filename)
    if report_name:
        with open(report_name, "r") as f:
            shutil.copyfileobj(f, outfile)
        return
    elif attempts_remaining > 0:
        # As the number of attempts remaining decreases, increase the number of seconds waited
        # if the max wait time is 2s and there are 10 attempts remaining, wait .2 seconds.
        # if the max wait time is 2s and there are 2 attempts remaining, wait 1 second.
        time.sleep(max_wait_time / attempts_remaining)
        output_report_with_retries(
            outfile, pid, filename, attempts_remaining - 1, max_wait_time
        )
    else:
        raise RuntimeError("Report not found for ({}, {}).".format(filename, pid))


def find_report_in_cur_dir(pid, filename):
    for report_name in sorted(glob.glob("{}_*.crash".format(filename)), reverse=True):
        # parse out pid from first line of report
        # `Process:               filename [pid]``
        with open(report_name) as cur_report:
            pattern = re.compile(r"Process: *{} \[([0-9]*)\]".format(filename))
            cur_report_pid = pattern.search(cur_report.readline()).group(1)

        assert cur_report_pid and cur_report_pid.isdigit()
        if cur_report_pid == pid:
            return report_name

    # did not find the crash report
    return None


if __name__ == "__main__":
    main()
