import json
import multiprocessing
import os
import re
import subprocess
import sys


def run_analyzer(data):
    os.chdir(data["directory"])
    command = (
        data["command"]
        + f" --analyze --analyzer-output html -o analyzer-results -Xclang -analyzer-config -Xclang max-nodes=75000"
    )
    print(command)
    subprocess.run(command, shell=True, check=True)


def pool_error(e):
    print("Error analyzing file:", e)


def main():
    db_path = sys.argv[1]
    database = json.load(open(db_path))

    with multiprocessing.Pool() as pool:
        pool.map_async(run_analyzer, [k for k in database], error_callback=pool_error)
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
