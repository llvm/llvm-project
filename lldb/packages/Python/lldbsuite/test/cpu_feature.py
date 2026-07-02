"""
Platform-agnostic helper to query for CPU features.
"""

import re


PF_ARM_SVE_INSTRUCTIONS_AVAILABLE = 46


class CPUFeature:
    def __init__(
        self,
        linux_cpu_info_flag: str = None,
        darwin_sysctl_key: str = None,
        windows_processor_feature: int = None,
    ):
        self.cpu_info_flag = linux_cpu_info_flag
        self.sysctl_key = darwin_sysctl_key
        self.windows_processor_feature = windows_processor_feature

    def __str__(self):
        for arch_class in ALL_ARCHS:
            for feat_var in dir(arch_class):
                if self == getattr(arch_class, feat_var):
                    return f"{arch_class.__name__}.{feat_var}"
        raise AssertionError("unreachable")

    def is_supported(self, triple, cmd_runner):
        if re.match(".*-.*-linux", triple):
            err_msg, res = self._is_supported_linux(cmd_runner)
        elif re.match(".*-apple-.*", triple):
            err_msg, res = self._is_supported_darwin(cmd_runner)
        elif re.match(".*-windows-.*", triple):
            err_msg, res = self._is_supported_windows(cmd_runner)
        else:
            err_msg, res = None, False

        if err_msg:
            print(f"CPU feature check failed: {err_msg}")

        return res

    def _is_supported_linux(self, cmd_runner):
        if not self.cpu_info_flag:
            return f"Unspecified cpuinfo flag for {self}", False

        cmd = "cat /proc/cpuinfo"
        err, retcode, output = cmd_runner(cmd)
        if err.Fail() or retcode != 0:
            return output, False

        # Assume that every processor presents the same features.
        # Look for the first "Features: ...." line. Features are space separated.
        if m := re.search(r"Features\s*: (.*)\n", output):
            features = m.group(1).split()
            return None, (self.cpu_info_flag in features)

        return 'No "Features:" line found in /proc/cpuinfo', False

    def _is_supported_darwin(self, cmd_runner):
        if not self.sysctl_key:
            return f"Unspecified sysctl key for {self}", False

        cmd = f"sysctl -n {self.sysctl_key}"
        err, retcode, output = cmd_runner(cmd)
        if err.Fail() or retcode != 0:
            return output, False

        return None, (output.strip() == "1")

    # PowerShell may not be on PATH on minimal Windows images, and Add-Type
    # requires the .NET CLR and the CSC compiler to be available. Neither is
    # guaranteed.
    # TODO: Replace the PowerShell chain with a probe that calls
    # 'IsProcessorFeaturePresent' directly.
    def _is_supported_windows(self, cmd_runner):
        import base64

        if self.windows_processor_feature is None:
            return f"Unspecified processor feature ID for {self}", False

        # IsProcessorFeaturePresent() via PowerShell
        ps_script = (
            "Add-Type -TypeDefinition '"
            "using System; using System.Runtime.InteropServices; "
            "public class WinAPI { "
            '[DllImport("kernel32.dll")] '
            "public static extern bool IsProcessorFeaturePresent(uint f); }'; "
            f"[WinAPI]::IsProcessorFeaturePresent({self.windows_processor_feature})"
        )

        # PowerShell -EncodedCommand expects UTF-16LE Base64.
        encoded = base64.b64encode(ps_script.encode("utf-16-le")).decode("ascii")
        cmd = f"powershell -EncodedCommand {encoded}"
        err, retcode, output = cmd_runner(cmd)
        if err.Fail() or retcode != 0:
            return (
                "Windows SVE detection via PowerShell failed "
                "(retcode={0}, output={1!r})".format(retcode, output)
            ), False

        return None, (output.strip().lower() == "true")


class AArch64:
    FPMR = CPUFeature("fpmr")
    POE = CPUFeature("poe")
    GCS = CPUFeature("gcs")
    MTE = CPUFeature("mte", "hw.optional.arm.FEAT_MTE4")
    MTE_STORE_ONLY = CPUFeature("mtestoreonly")
    PTR_AUTH = CPUFeature("paca", "hw.optional.arm.FEAT_PAuth2")
    SME = CPUFeature("sme", "hw.optional.arm.FEAT_SME")
    SME_FA64 = CPUFeature("smefa64")
    SME2 = CPUFeature("sme2", "hw.optional.arm.FEAT_SME2")
    SVE = CPUFeature("sve", windows_processor_feature=PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)


class Loong:
    LASX = CPUFeature("lasx")
    LSX = CPUFeature("lsx")


ALL_ARCHS = [AArch64, Loong]
