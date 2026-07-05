if config.root.host_arch not in ["AMD64", "x86_64"]:
    config.unsupported = True

if config.target_arch not in ["AMD64", "x86_64"]:
    config.unsupported = True
