def getRoot(config):
    if not config.parent:
        return config
    return getRoot(config.parent)


root = getRoot(config)

if root.host_os not in ["Windows"]:
    config.unsupported = True
