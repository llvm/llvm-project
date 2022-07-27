#!/usr/bin/python3

from git import Repo
import re
import sys

version = sys.argv[1]

repo = Repo()

tag = repo.git.describe(tags = True, abbrev=0)
m = re.match('llvmorg-([0-9]+)\.([0-9]+)\.([0-9]+)', tag)

if m:
    expected_major = m.group(1)
    expected_minor = m.group(2)
    expected_patch = int(m.group(3)) + 1
else:
    # If the previous tag is llvmorg-X-init, then we should be at version X.0.0.
    m = re.match('llvmorg-([0-9]+)-init', tag)
    if not m:
        print("error: Tag is not valid: ", tag)
        sys.exit(1)
    expected_major = m.group(1)
    expected_minor = 0
    expected_patch = 0

expected_version = f"{expected_major}.{expected_minor}.{expected_patch}"

m = re.match("[0-9]+\.[0-9]+\.[0-9]+", version)
if not m:
    print("error: Version is not valid: ", version)
    sys.exit(1)

if version != expected_version:
    print("error: Expected version", expected_version, "but found version", version)
    sys.exit(1)

print("Versions match:", version, expected_version)
sys.exit(0)
