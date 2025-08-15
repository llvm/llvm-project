"""
Platform-agnostic helper to query for CPU features.
"""

import re


class CPUFeature:
    def __init__(self, linux_cpu_info_flag: str, darwin_sysctl_key: str):
        self.cpu_info_flag = linux_cpu_info_flag
        self.sysctl_key = darwin_sysctl_key

    def is_supported(self, triple, cmd_runner):
        if re.match(".*-.*-linux", triple):
            err_msg, res = self._is_supported_linux(cmd_runner)
        elif re.match(".*-apple-.*", triple):
            err_msg, res = self._is_supported_darwin(cmd_runner)
        else:
            err_msg, res = None, False

        if err_msg:
            print(f"CPU feature check failed: {err_msg}")

        return res

    def _is_supported_linux(self, cmd_runner):
        cmd = "cat /proc/cpuinfo"
        err, retcode, output = cmd_runner(cmd)
        if err.Fail() or retcode != 0:
            err_msg = f"cat /proc/cpuinfo failed: {output}"
            return err_msg, False

        return None, (self.cpu_info_flag in output)

    def _is_supported_darwin(self, cmd_runner):
        cmd = f"sysctl -n {self.sysctl_key}"
        err, retcode, output = cmd_runner(cmd)
        if err.Fail() or retcode != 0:
            return output, False

        return None, (output.strip() == "1")


# List of CPU features
FPMR = CPUFeature("fpmr", "???")
GCS = CPUFeature("gcs", "???")
LASX = CPUFeature("lasx", "???")
LSX = CPUFeature("lsx", "???")
MTE = CPUFeature("mte", "???")
MTE_STORE_ONLY = CPUFeature("mtestoreonly", "???")
PTR_AUTH = CPUFeature("paca", "hw.optional.arm.FEAT_PAuth2")
SME = CPUFeature("sme", "hw.optional.arm.FEAT_SME")
SME_FA64 = CPUFeature("smefa64", "???")
SME2 = CPUFeature("sme2", "hw.optional.arm.FEAT_SME2")
SVE = CPUFeature("sve", "???")
