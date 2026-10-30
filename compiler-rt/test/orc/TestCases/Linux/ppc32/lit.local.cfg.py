if config.root.host_arch not in ["ppc", "powerpc"]:
    config.unsupported = True

if config.target_arch != "powerpc":
    config.unsupported = True
