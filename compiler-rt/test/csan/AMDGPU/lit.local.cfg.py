# Discover only tests for GPU frontends that are usable on this system.
config.suffixes = []
if "csan-hip" in config.available_features:
    config.suffixes.append(".hip")
if "csan-openmp-offload" in config.available_features:
    config.suffixes.append(".cpp")

if not config.suffixes:
    config.unsupported = True
else:
    config.parallelism_group = "gpu"
