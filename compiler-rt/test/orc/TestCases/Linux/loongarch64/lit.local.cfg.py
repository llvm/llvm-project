if config.root.host_arch != "loongarch64":
    config.unsupported = True

if config.target_arch != "loongarch64":
    config.unsupported = True
