if config.root.host_arch != "s390x":
    config.unsupported = True

if config.target_arch != "s390x":
    config.unsupported = True
