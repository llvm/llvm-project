import sys
import shutil

with open(sys.argv[1] + ".output") as log_file:
    lines = log_file.readlines()
    for line in lines:
        if line.startswith(sys.argv[2]):
            pid = line.split(" ")[1].strip()
            shutil.copy(
                sys.argv[1] + "." + pid + ".fdata",
                sys.argv[1] + "." + sys.argv[3] + ".fdata",
            )
            sys.exit(0)

sys.exit(1)
