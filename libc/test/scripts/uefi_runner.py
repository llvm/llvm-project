#!/usr/bin/env python3
#
# ===- UEFI runner for binaries  ------------------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import argparse
import os
import platform
import re
import shutil
import subprocess
import tempfile


class Target:
    def __init__(self, triple: str):
        self.triple = triple.split("-")
        assert len(self.triple) == 2 or len(self.triple) == 3

    def arch(self):
        return self.triple[0]

    def isNativeArch(self):
        return self.arch() == Target.defaultArch()

    def vendor(self):
        if len(self.triple) == 2:
            return "unknown"

        return self.triple[1]

    def os(self):
        if len(self.triple) == 2:
            return self.triple[1]

        return self.triple[2]

    def abi(self):
        if len(self.triple) < 4:
            return "llvm"

        return self.triple[3]

    def qemuBinary(self):
        return f"qemu-system-{self.arch()}"

    def qemuArgs(self):
        if self.arch() == "aarch64":
            args = ["-machine", "virt"]

            if self.isNativeArch():
                args.pop()
                args.append("virt,gic-version=max,accel=kvm:tcg")
                args.append("-cpu")
                args.append("max")

            return args

        if self.arch() == "x86_64" and self.isNativeArch():
            return [
                "-machine",
                "accel=kvm:tcg",
                "-cpu",
                "max",
            ]
        return []

    def ovmfPath(self):
        if self.arch() == "aarch64":
            return "AAVMF_CODE.fd"

        if self.arch() == "x86_64":
            return "OVMF_CODE.fd"

        raise Exception(f"{self.arch()} is not a valid architecture")

    def efiArch(self):
        if self.arch() == "aarch64":
            return "AA64"

        if self.arch() == "x86_64":
            return "X64"

        raise Exception(f"{self.arch()} is not a valid architecture")

    def efiFileName(self):
        return f"BOOT{self.efiArch()}.EFI"

    def __str__(self):
        return f"{self.arch()}-{self.vendor()}-{self.os()}-{self.abi()}"

    def default():
        return Target(f"{Target.defaultArch()}-unknown-{Target.defaultOs()}")

    def defaultArch():
        return platform.machine()

    def defaultOs():
        return platform.system().lower()


def main():
    parser = argparse.ArgumentParser(description="UEFI runner for binaries")
    parser.add_argument("binary_file", help="Path to the UEFI binary to execute")
    parser.add_argument(
        "--target",
        help="Triplet which specifies what the target is",
    )
    parser.add_argument(
        "--ovmf-path",
        help="Path to the directory where OVMF is located",
    )
    args = parser.parse_args()
    target = Target.default() if args.target is None else Target(args.target)

    ovmfFile = os.path.join(
        args.ovmf_path
        or os.getenv("OVMF_PATH")
        or f"/usr/share/edk2/{target.efiArch().lower()}",
        target.ovmfPath(),
    )

    qemuArgs = [target.qemuBinary()]
    qemuArgs.extend(target.qemuArgs())

    qemuArgs.append("-drive")
    qemuArgs.append(f"if=pflash,format=raw,unit=0,readonly=on,file={ovmfFile}")

    qemuArgs.append("-nographic")
    qemuArgs.append("-serial")
    qemuArgs.append("stdio")

    qemuArgs.append("-monitor")
    qemuArgs.append("none")

    with tempfile.TemporaryDirectory() as tempdir:
        qemuArgs.append("-drive")
        qemuArgs.append(f"file=fat:rw:{tempdir},format=raw,media=disk")

        os.mkdir(os.path.join(tempdir, "EFI"))
        os.mkdir(os.path.join(tempdir, "EFI", "BOOT"))

        shutil.copyfile(
            args.binary_file, os.path.join(tempdir, "EFI", "BOOT", target.efiFileName())
        )

        proc = subprocess.Popen(
            qemuArgs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        num_tests = 0
        num_suites = 0

        while True:
            line = proc.stdout.readline()
            if not line:
                break

            line = line.rstrip()

            if num_tests > 0:
                print(line)

            x = re.search(r"Running ([0-9]+) tests? from ([0-9]+) tests? suite\.", line)
            if not x is None:
                num_tests = int(x.group(1))
                num_suites = int(x.group(2))
                continue

            x = re.search(
                r"Ran ([0-9]+) tests?\.  PASS: ([0-9]+)  FAIL: ([0-9]+)", line
            )

            if not x is None:
                proc.kill()
                ran_tests = int(x.group(1))
                passed_tests = int(x.group(2))
                failed_tests = int(x.group(3))

                assert passed_tests + failed_tests == ran_tests
                assert ran_tests == num_tests

                if failed_tests > 0:
                    raise Exception("A test failed")
                break


if __name__ == "__main__":
    main()
