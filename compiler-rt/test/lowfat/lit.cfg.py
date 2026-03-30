import lit.formats
import os

# Setup config name.
config.name = "LowFatSanitizer" + getattr(config, "name_suffix", "")

# Setup source root.
config.test_source_root = os.path.dirname(__file__)
config.suffixes = [".c", ".cpp"]

# Teach lit that these are shell tests (// RUN: ... lines).
# When loaded via the build-dir site config, lit.common.configured sets this;
# when invoked directly against the source dir, we must set it ourselves.
if not hasattr(config, "test_format"):
    config.test_format = lit.formats.ShTest(execute_external=True)


# Find clang++:
#   1. config.clang is set by lit.common.configured (when run via build-dir site config)
#   2. config.llvm_tools_dir is set by lit.common.configured
#   3. Fall back to bare "clang++" on PATH for direct source-dir invocations
def _find_clang():
    clang = getattr(config, "clang", None)
    if clang:
        return clang
    tools_dir = getattr(config, "llvm_tools_dir", None)
    if tools_dir:
        candidate = os.path.join(tools_dir, "clang++")
        if os.path.isfile(candidate):
            return candidate
    return "clang++"


clang = _find_clang()


def build_invocation(flags):
    return " " + " ".join([clang] + flags) + " "


# Base flags
lowfat_base = ["-fsanitize=lowfat"]

# safe mode (fast mode is the default)
lowfat_safe = lowfat_base + ["-mllvm", "-lowfat-mode=safe"]

# right-align mode: allocations are biased toward the high end of the slot
# while preserving the platform's default malloc alignment.
lowfat_right_align = lowfat_base + ["-mllvm", "-lowfat-mode=right-align"]

config.substitutions.append(("%clangxx_lowfat ", build_invocation(lowfat_base)))
config.substitutions.append(("%clangxx_lowfat_safe ", build_invocation(lowfat_safe)))
config.substitutions.append(("%clangxx_lowfat_right_align ", build_invocation(lowfat_right_align)))

# Recover mode versions
config.substitutions.append(
    (
        "%clangxx_lowfat_recover ",
        build_invocation(lowfat_base + ["-fsanitize-recover=lowfat"]),
    )
)
config.substitutions.append(
    (
        "%clangxx_lowfat_safe_recover ",
        build_invocation(lowfat_safe + ["-fsanitize-recover=lowfat"]),
    )
)

# Only Darwin and Linux are supported.
if getattr(config, "target_os", "Unknown") not in ["Darwin", "Linux"]:
    # When running directly (no site config), target_os may be unset; don't
    # mark unsupported in that case — let the tests fail naturally if the
    # platform truly isn't supported.
    if hasattr(config, "target_os"):
        config.unsupported = True

# Expose 'lowfat-custom-config' feature when the runtime was built with a
# custom sizes.cfg (i.e. -DLOWFAT_SIZES_CFG was set at cmake time).
# Tests guarded with REQUIRES: lowfat-custom-config are skipped otherwise.
if getattr(config, "lowfat_custom_config", False):
    config.available_features.add("lowfat-custom-config")
