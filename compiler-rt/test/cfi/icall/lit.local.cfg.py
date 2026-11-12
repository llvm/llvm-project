# The cfi-icall checker is only supported on x86 and x86_64 for now.
if config.root.host_arch not in ["x86", "x86_64", "loongarch64"]:
    config.unsupported = True
