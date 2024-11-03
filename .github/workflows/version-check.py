#!/usr/bin/python3

from git import Repo
import re
import sys


def get_version_from_tag(tag):
    m = re.match('llvmorg-([0-9]+)\.([0-9]+)\.([0-9]+)(-rc[0-9]+)?$', tag)
    if m:
        if m.lastindex == 4:
            # We have an rc tag.
            return m.group(1,2,3)
        # We have a final release tag.
        return (m.group(1), m.group(2), str(int(m.group(3)) + 1))

    m = re.match('llvmorg-([0-9]+)-init', tag)
    if m:
        return (m.group(1), "0", "0")

    raise Exception(f"error: Tag is not valid: {tag}")


version = sys.argv[1]

repo = Repo()

tag = repo.git.describe(tags = True, abbrev=0)
expected_version = '.'.join(get_version_from_tag(tag))

if version != expected_version:
    print("error: Expected version", expected_version, "but found version", version)
    sys.exit(1)

print("Versions match:", version, expected_version)
sys.exit(0)
