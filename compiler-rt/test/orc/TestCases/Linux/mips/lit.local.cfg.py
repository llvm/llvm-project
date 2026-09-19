if config.root.host_arch not in ["mips", "mipsel", "mips64", "mips64el"]:
    config.unsupported = True

if config.target_arch not in ["mips", "mipsel", "mips64", "mips64el"]:
    config.unsupported = True
