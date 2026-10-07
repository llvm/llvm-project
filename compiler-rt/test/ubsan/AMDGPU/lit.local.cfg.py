# These tests require a functioning GPU device and runtime on the system.
if "ubsan-standalone" not in config.available_features or (
    "ubsan-hip" not in config.available_features
    and "ubsan-openmp-offload" not in config.available_features
):
    config.unsupported = True
else:
    config.parallelism_group = "gpu"
    config.suffixes = [".c", ".cpp", ".hip"]
